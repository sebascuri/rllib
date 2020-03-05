"""Python Script Template."""

from collections import namedtuple
import torch
import torch.distributions
import torch.nn as nn
import copy
from rllib.util.neural_networks import freeze_parameters, repeat_along_dimension
from rllib.util.utilities import separated_kl

MPOLosses = namedtuple('MPOLosses', ('policy_loss', 'eta_loss'))
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

    """

    def __init__(self, epsilon: float, epsilon_mean: float, epsilon_var: float):
        super().__init__()

        self.eta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_mean = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_var = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.epsilon = torch.tensor(epsilon)  # type: Tensor
        self.epsilon_mean = epsilon_mean
        self.epsilon_var = epsilon_var




class ModelBasedMPO(nn.Module):
    """MPO Algorithm based on a system model.

    This method uses the `MPOLoss`, but constructs the Q-function using the value
    function together with the model.

    Reference: https://arxiv.org/abs/1806.06920

    Parameters
    ----------
    model : callable
    reward_function : callable
    policy : nn.Module
    value_function : nn.Module
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

    # __constants__ = ['epsilon', 'epsilon_mean', 'epsilon_var']

    def __init__(self, model, reward_function, policy, value_function,
                 epsilon: float, epsilon_mean: float, epsilon_var: float, gamma: float):

        old_policy = copy.deepcopy(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.eta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_mean = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.eta_var = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.old_policy = old_policy
        self.model = model
        self.reward_function = reward_function
        self.policy = policy
        self.value_function = value_function
        self.gamma = gamma

        self.epsilon = torch.tensor(epsilon)
        self.epsilon_mean = torch.tensor(epsilon_mean)
        self.epsilon_var = torch.tensor(epsilon_var)

        self.value_loss = nn.MSELoss(reduction='mean')

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def project_etas(self):
        """Project the etas to be positive inplace."""
        # Since we divide by eta, make sure it doesn't go to zero.
        self.eta.data.clamp_(1e-5, torch.finfo(self.eta.data.dtype).max)
        self.eta_mean.data.clamp_(1e-5, torch.finfo(self.eta_mean.data.dtype).max)
        self.eta_var.data.clamp_(1e-5, torch.finfo(self.eta_var.data.dtype).max)

    def mpo_loss(self, q_values, action_log_probs, kl_mean, kl_var):
        """Return value targets and loss terms from MPO.

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
        # Make sure the lagrange multiplies stays positive
        self.project_etas()

        # E-step: Created a weighed, sample-based representation of the optimal policy
        q_values = q_values.detach() * (torch.tensor(1.) / self.eta)
        normalizer = torch.logsumexp(q_values, dim=0)
        num_actions = torch.tensor(action_log_probs.shape[0]).float()
        dual_loss = self.eta * (
                    self.epsilon + torch.mean(normalizer) - torch.log(num_actions))
        weights = torch.exp(q_values - normalizer.detach())

        # M-step: Fit the parametric policy to the representation form the E-step
        weighted_log_prob = torch.sum(weights * action_log_probs, dim=0)
        log_likelihood = torch.mean(weighted_log_prob)

        kl_loss = self.eta_mean.detach() * kl_mean + self.eta_var.detach() * kl_var
        policy_loss = -log_likelihood + kl_loss

        eta_mean_loss = self.eta_mean * (self.epsilon_mean - kl_mean.detach())
        eta_var_loss = self.eta_var * (self.epsilon_var - kl_var.detach())

        dual_losses = dual_loss + eta_mean_loss + eta_var_loss

        return MPOLosses(policy_loss, dual_losses)

    def forward(self, states, num_action_samples):
        """Compute the losses for one step of MPO.

        Parameters
        ----------
        states : torch.Tensor
            The states at which to compute the losses.
        num_action_samples : int
            The number of actions to sample.

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
        pi_dist = self.policy(states)
        pi_dist_old = self.old_policy(states)
        value_prediction = self.value_function(states)

        kl_mean, kl_var = separated_kl(p=pi_dist, q=pi_dist_old)

        actions = pi_dist.sample((num_action_samples,))
        states = repeat_along_dimension(states, num_action_samples, dim=0)
        action_log_probs = pi_dist.log_prob(actions)

        # Compute q-values and values using the model
        with torch.no_grad():
            next_states = self.model(states, actions).sample()
            reward = self.reward_function(states, actions)
            q_values = reward + self.gamma * self.value_function(next_states)

        # Since actions come from policy, value is the expected q-value
        value_loss = self.value_loss(value_prediction, q_values.mean(dim=0))

        losses = self.mpo_loss(q_values=q_values,
                               action_log_probs=action_log_probs,
                               kl_mean=kl_mean,
                               kl_var=kl_var)

        combined_loss = value_loss + losses.policy_loss.mean() + losses.eta_loss.mean()

        return MPOReturn(loss=combined_loss,
                         value_loss=value_loss,
                         policy_loss=losses.policy_loss,
                         eta_loss=losses.eta_loss,
                         kl_div=kl_mean + kl_var)
