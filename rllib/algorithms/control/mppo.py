"""Python Script Template."""

from collections import namedtuple

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from tqdm import tqdm

from rllib.algorithms.dyna import dyna_rollout
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks import freeze_parameters
from rllib.util.rollout import rollout_model
from rllib.util.utilities import separated_kl, tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module

MPOLosses = namedtuple('MPOLosses', ['primal_loss', 'dual_loss'])
MPOReturn = namedtuple('MPOReturn', ['loss', 'value_loss', 'policy_loss', 'eta_loss',
                                     'kl_div'])


class MPPO(nn.Module):
    """Maximum a Posterior Policy Optimization.

    This method uses critic values under samples from a policy to construct a
    sample-based representation of the optimal policy. It then fits the parametric
    policy to this representation via supervised learning.

    Parameters
    ----------
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean: float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.
    epsilon_var: float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.

    References
    ----------
    Abdolmaleki, et al. "Maximum a Posteriori Policy Optimisation." (2018). ICLR.
    """

    __constants__ = ['epsilon', 'epsilon_mean', 'epsilon_var']

    def __init__(self, epsilon, epsilon_mean, epsilon_var):
        super().__init__()

        self.eta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_mean = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_var = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.epsilon = torch.tensor(epsilon)
        self.epsilon_mean = torch.tensor(epsilon_mean)
        self.epsilon_var = torch.tensor(epsilon_var)

    def project_etas(self):
        """Project the etas to be positive inplace."""
        # Since we divide by eta, make sure it doesn't go to zero.
        self.eta.data.clamp_(min=1e-5)
        self.eta_mean.data.clamp_(min=1e-5)
        self.eta_var.data.clamp_(min=1e-5)

    def forward(self, q_values, action_log_probs, kl_mean, kl_var):
        """Return primal and dual loss terms from MMPO.

        Parameters
        ----------
        q_values : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of values for
            state-action pairs.
        action_log_probs : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of log probabilities
            of the corresponding actions under the policy.
        kl_mean : torch.Tensor
            A float corresponding to the KL divergence.
        kl_var : torch.Tensor
            A float corresponding to the KL divergence.
        """
        # Make sure the lagrange multipliers stay positive.
        self.project_etas()

        # E-step: Solve Problem (7).
        # Create a weighed, sample-based representation of the optimal policy q Eq(8).
        # Compute the dual loss for the constraint KL(q || old_pi) < eps.
        q_values = q_values.detach() * (torch.tensor(1.) / self.eta)
        normalizer = torch.logsumexp(q_values, dim=0)
        num_actions_ = action_log_probs.shape[0]
        num_actions = torch.tensor(num_actions_, dtype=torch.get_default_dtype())

        dual_loss = self.eta * (
                self.epsilon + torch.mean(normalizer) - torch.log(num_actions))
        # non-parametric representation of the optimal policy.
        weights = torch.exp(q_values - normalizer.detach())

        # M-step: # E-step: Solve Problem (10).
        # Fit the parametric policy to the representation form the E-step.
        # Maximize the log_likelihood of the weighted log probabilities, subject to the
        # KL divergence between the old_pi and the new_pi to be smaller than epsilon.

        weighted_log_prob = torch.sum(weights * action_log_probs, dim=0)
        log_likelihood = torch.mean(weighted_log_prob)

        kl_loss = self.eta_mean.detach() * kl_mean + self.eta_var.detach() * kl_var
        primal_loss = -log_likelihood + kl_loss

        eta_mean_loss = self.eta_mean * (self.epsilon_mean - kl_mean.detach())
        eta_var_loss = self.eta_var * (self.epsilon_var - kl_var.detach())

        dual_loss = dual_loss + eta_mean_loss + eta_var_loss

        return MPOLosses(primal_loss, dual_loss)


class MBMPPO(nn.Module):
    """MPPO Algorithm based on a system model.

    This method uses the `MPPOLoss`, but constructs the Q-function using the value
    function together with the model.

    Parameters
    ----------
    dynamical_model : AbstractModel
    reward_model : AbstractReward
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    epsilon : float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean : float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    epsilon_var : float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    gamma : float
        The discount factor.
    """

    def __init__(self, dynamical_model, reward_model, policy, value_function,
                 epsilon, epsilon_mean, epsilon_var, gamma, num_action_samples=15,
                 entropy_reg=0.,
                 termination=None):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.policy = policy
        self.value_function = value_function
        self.gamma = gamma

        self.mppo = MPPO(epsilon, epsilon_mean, epsilon_var)
        self.value_loss = nn.MSELoss(reduction='mean')
        self.num_action_samples = num_action_samples
        self.entropy_reg = entropy_reg
        self.termination = termination

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def forward(self, states):
        """Compute the losses for one step of MPO.

        Parameters
        ----------
        states : torch.Tensor
            The states at which to compute the losses.

        Returns
        -------
        loss : torch.Tensor
            The combined loss
        value_loss : torch.Tensor
            The loss of the value function approximation.
        policy_loss : torch.Tensor
            The kl-regularized fitting loss for the policy.
        eta_loss : torch.Tensor
            The loss for the lagrange multipliers.
        kl_mean : torch.Tensor
            The average KL divergence of the mean.
        kl_var : torch.Tensor
            The average KL divergence of the variance.
        """
        value_prediction = self.value_function(states)

        pi_dist = tensor_to_distribution(self.policy(states))
        pi_dist_old = tensor_to_distribution(self.old_policy(states))
        kl_mean, kl_var = separated_kl(p=pi_dist, q=pi_dist_old)

        with torch.no_grad():
            dyna_return = dyna_rollout(state=states,
                                       model=self.dynamical_model, policy=self.policy,
                                       reward=self.reward_model, steps=0,
                                       gamma=self.gamma,
                                       value_function=self.value_function,
                                       num_samples=self.num_action_samples,
                                       entropy_reg=self.entropy_reg,
                                       termination=self.termination,
                                       )
        q_values = dyna_return.q_target
        action_log_probs = pi_dist.log_prob(dyna_return.trajectory[0].action)

        # Since actions come from policy, value is the expected q-value
        value_loss = self.value_loss(value_prediction, q_values.mean(dim=0))

        losses = self.mppo(q_values=q_values, action_log_probs=action_log_probs,
                           kl_mean=kl_mean, kl_var=kl_var)

        combined_loss = value_loss + losses.primal_loss.mean() + losses.dual_loss.mean()

        return MPOReturn(loss=combined_loss,
                         value_loss=value_loss,
                         policy_loss=losses.primal_loss,
                         eta_loss=losses.dual_loss,
                         kl_div=kl_mean + kl_var)


def train_mppo(mppo: MBMPPO, initial_distribution, optimizer,
               num_iter, num_trajectories, num_simulation_steps, refresh_interval,
               batch_size, num_subsample):
    """Train MPPO policy."""
    value_losses = []
    eta_parameters = []
    policy_losses = []
    policy_returns = []
    kl_div = []
    for i in tqdm(range(num_iter)):
        # Compute the state distribution
        if i % refresh_interval == 0:
            with torch.no_grad():
                initial_states = initial_distribution.sample((num_trajectories,))
                trajectory = rollout_model(mppo.dynamical_model,
                                           reward_model=mppo.reward_model,
                                           policy=mppo.policy,
                                           initial_state=initial_states,
                                           max_steps=num_simulation_steps)
                trajectory = Observation(*stack_list_of_tuples(trajectory))
                policy_returns.append(trajectory.reward.sum(dim=0).mean().item())

                # Shuffle to get a state distribution
                states = trajectory.state.reshape(-1, trajectory.state.shape[-1])
                np.random.shuffle(states.numpy())
                state_batches = torch.split(states, batch_size)[::num_subsample]

        policy_episode_loss = 0.
        value_episode_loss = 0.
        episode_kl_div = 0.

        # Copy over old policy for KL divergence
        mppo.reset()

        # Iterate over state batches in the state distribution
        for states in state_batches:
            optimizer.zero_grad()
            losses = mppo(states)
            losses.loss.backward()
            optimizer.step()

            # Track statistics
            value_episode_loss += losses.value_loss.item()
            policy_episode_loss += losses.policy_loss.item()
            episode_kl_div += losses.kl_div.item()

        value_losses.append(value_episode_loss / len(state_batches))
        policy_losses.append(policy_episode_loss / len(state_batches))
        eta_parameters.append(
            [eta.detach().clone().numpy() for name, eta in mppo.named_parameters() if
             'eta' in name])
        kl_div.append(episode_kl_div)

    return value_losses, policy_losses, policy_returns, eta_parameters, kl_div
