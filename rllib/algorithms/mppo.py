"""Maximum a Posterior Policy Optimization algorithm."""

from collections import namedtuple

import torch
import torch.distributions
import torch.nn as nn

from .abstract_algorithm import AbstractAlgorithm, MPOLoss

from rllib.util.neural_networks import freeze_parameters
from rllib.util.utilities import separated_kl, tensor_to_distribution, RewardTransformer
from rllib.util.value_estimation import mb_return
from rllib.util.neural_networks import deep_copy_module, repeat_along_dimension, \
    update_parameters
from rllib.util.parameter_decay import Learnable, Constant, ParameterDecay

MPOLosses = namedtuple('MPOLosses', ['primal_loss', 'dual_loss'])


class MPPOWorker(nn.Module):
    """Maximum a Posterior Policy Optimization Losses.

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

    def __init__(self, epsilon=None, epsilon_mean=None, epsilon_var=None,
                 eta=None, eta_mean=None, eta_var=None):
        super().__init__()

        assert (epsilon is not None) ^ (eta is not None), "XOR(eps, eta)."
        assert (epsilon_mean is not None) ^ (eta_mean is not None), "XOR(eps_m, eta_m)."

        if eta is not None:  # Regularization: \eta KL(q || \pi)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
            self.epsilon = torch.tensor(0.)
        else:  # Trust-Region: || KL(q || \pi_old) || < \epsilon
            self.eta = Learnable(1., positive=True)
            self.epsilon = torch.tensor(epsilon)

        if eta_mean is not None:  # Regularization: \eta_m KL_m(\pi_old || \pi)
            if not isinstance(eta_mean, ParameterDecay):
                eta_mean = Constant(eta_mean)
            self.eta_mean = eta_mean
            self.epsilon_mean = torch.tensor(0.)
        else:  # Trust-Region: || KL_m(\pi_old || \pi) || < \epsilon_m
            self.eta_mean = Learnable(1., positive=True)
            self.epsilon_mean = torch.tensor(epsilon_mean)

        if eta_var is not None:  # Regularization: \eta_var KL_var(\pi_old || \pi)
            if not isinstance(eta_var, ParameterDecay):
                eta_var = Constant(eta_var)
            self.eta_var = eta_var
            self.epsilon_var = torch.tensor(0.)
        elif epsilon_var is not None:  # Trust-Region:
            self.eta_var = Learnable(1., positive=True)
            self.epsilon_var = torch.tensor(epsilon_var)
        else:  # KL-DIV not separated into mean and var components.
            self.eta_var = Learnable(1., positive=True)
            self.epsilon_var = torch.tensor(0.)

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
        # self.project_etas()

        # E-step: Solve Problem (7).
        # Create a weighed, sample-based representation of the optimal policy q Eq(8).
        # Compute the dual loss for the constraint KL(q || old_pi) < eps.
        q_values = q_values.detach() * (torch.tensor(1.) / self.eta())
        normalizer = torch.logsumexp(q_values, dim=0)
        num_actions = torch.tensor(1. * action_log_probs.shape[0])

        dual_loss = self.eta() * (
                self.epsilon + torch.mean(normalizer) - torch.log(num_actions))
        # non-parametric representation of the optimal policy.
        weights = torch.exp(q_values - normalizer.detach())

        # M-step: # E-step: Solve Problem (10).
        # Fit the parametric policy to the representation form the E-step.
        # Maximize the log_likelihood of the weighted log probabilities, subject to the
        # KL divergence between the old_pi and the new_pi to be smaller than epsilon.

        weighted_log_prob = torch.sum(weights * action_log_probs, dim=0)
        log_likelihood = torch.mean(weighted_log_prob)

        kl_loss = self.eta_mean().detach() * kl_mean + self.eta_var().detach() * kl_var
        primal_loss = -log_likelihood + kl_loss

        eta_mean_loss = self.eta_mean() * (self.epsilon_mean - kl_mean.detach())
        eta_var_loss = self.eta_var() * (self.epsilon_var - kl_var.detach())

        dual_loss = dual_loss + eta_mean_loss + eta_var_loss

        return MPOLosses(primal_loss, dual_loss)


class MPPO(AbstractAlgorithm):
    """Maximum a Posteriori Policy Optimizaiton.

    The MPPO algorithm returns a loss that is a combination of three losses.

    - The dual loss associated with the variational distribution (Eq. 9)
    - The dual loss associated with the KL-hard constraint (Eq. 12).
    - The primal loss associated with the policy fitting term (Eq. 12).
    - A policy evaluation loss (Eq. 13).

    To compute the primal and dual losses, it uses the MPPOLoss module.

    Parameters
    ----------
    policy : AbstractPolicy
    q_function : AbstractQFunction
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean: float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    epsilon_var: float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    num_action_samples: int.
        Number of action samples to approximate integral.
    gamma: float
        The discount factor.

    References
    ----------
    Abdolmaleki, et al. (2018)
    Maximum a Posteriori Policy Optimisation. ICLR.

    TODO: Add Retrace for policy evaluation.
    """

    def __init__(self, policy, q_function, num_action_samples, criterion,
                 entropy_reg=0., epsilon=None, epsilon_mean=None, epsilon_var=None,
                 eta=None, eta_mean=None, eta_var=None, gamma=0.99,
                 reward_transformer=RewardTransformer):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.policy = policy
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.num_action_samples = num_action_samples
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.reward_transformer = reward_transformer

        self.mppo_loss = MPPOWorker(epsilon, epsilon_mean, epsilon_var,
                                    eta, eta_mean, eta_var)
        self.value_loss = criterion(reduction='mean')

    def get_kl_and_pi(self, state):
        """Get kl divergence and current policy at a given state.

        Compute the separated KL divergence between current and old policy.
        When the policy is a MultivariateNormal distribution, it compute the divergence
        that correspond to the mean and the covariance separately.

        When the policy is a Categorical distribution, it computes the divergence and
        assigns it to the mean component. The variance component is kept to zero.

        Parameters
        ----------
        state: torch.Tensor
            Empirical state distribution.

        Returns
        -------
        kl_mean: torch.Tensor
            KL-Divergence due to the change in the mean between current and
            previous policy.

        kl_var: torch.Tensor
            KL-Divergence due to the change in the variance between current and
            previous policy.

        pi_dist: torch.distribution.Distribution
            Current policy distribution.
        """
        pi_dist = tensor_to_distribution(self.policy(state, normalized=True))
        pi_dist_old = tensor_to_distribution(self.old_policy(state, normalized=True))

        if isinstance(pi_dist, torch.distributions.MultivariateNormal):
            kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)
        else:
            kl_mean = torch.distributions.kl_divergence(p=pi_dist_old, q=pi_dist).mean()
            kl_var = torch.zeros_like(kl_mean)

        return kl_mean, kl_var, pi_dist

    def forward(self, state, action, reward, next_state, done):
        """Compute the losses for one step of MPPO.

        Parameters
        ----------
        state: torch.Tensor
            The state at which to compute the losses.
        action: torch.Tensor
            Sampled action.
        reward: torch.Tensor
            Sampled reward.
        next_state: torch.Tensor
            Sampled next state.
        done: torch.Tensor
            Sampled done flag.

        Returns
        -------
        loss: torch.Tensor
            The combined loss
        value_loss: torch.Tensor
            The loss of the value function approximation.
        policy_loss: torch.Tensor
            The kl-regularized fitting loss for the policy.
        eta_loss: torch.Tensor
            The loss for the lagrange multipliers.
        kl_div: torch.Tensor
            The average KL divergence of the policy.
        """
        value_pred = self.q_function(state, action / self.policy.action_scale)
        state = repeat_along_dimension(state, number=self.num_action_samples, dim=0)
        next_state = repeat_along_dimension(next_state, number=self.num_action_samples,
                                            dim=0)

        kl_mean, kl_var, pi_dist = self.get_kl_and_pi(state)

        sampled_action = pi_dist.sample()
        action_log_probs = pi_dist.log_prob(sampled_action)

        losses = self.mppo_loss(q_values=self.q_target(state, sampled_action),
                                action_log_probs=action_log_probs,
                                kl_mean=kl_mean,
                                kl_var=kl_var)

        with torch.no_grad():
            next_pi = tensor_to_distribution(self.old_policy(next_state,
                                                             normalized=True))
            next_action = next_pi.sample()

            next_values = self.q_target(next_state, next_action) * (1. - done)
            value_target = self.reward_transformer(reward) + self.gamma * next_values

        value_loss = self.value_loss(value_pred, value_target.mean(dim=0))
        td_error = value_pred - value_target.mean(dim=0)

        dual_loss = losses.dual_loss.mean()
        policy_loss = losses.primal_loss.mean()
        combined_loss = value_loss + dual_loss + policy_loss

        self._info = {'kl_div': kl_mean + kl_var, 'kl_mean': kl_mean, 'kl_var': kl_var,
                      'eta': self.mppo_loss.eta(),
                      'eta_mean': self.mppo_loss.eta_mean(),
                      'eta_var': self.mppo_loss.eta_var()
                      }

        return MPOLoss(loss=combined_loss, dual=dual_loss, policy_loss=policy_loss,
                       critic_loss=value_loss, td_error=td_error)

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self):
        """Update target networks."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class MBMPPO(AbstractAlgorithm):
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
                 criterion, epsilon=None, epsilon_mean=None, epsilon_var=None,
                 eta=None, eta_mean=None, eta_var=None, gamma=0.99,
                 num_action_samples=15, entropy_reg=0.,
                 reward_transformer=RewardTransformer(), termination=None):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.policy = policy
        self.value_function = value_function
        self.value_target = deep_copy_module(value_function)
        self.gamma = gamma

        self.mppo_loss = MPPOWorker(epsilon, epsilon_mean, epsilon_var,
                                    eta, eta_mean, eta_var)
        self.value_loss = criterion(reduction='mean')

        self.num_action_samples = num_action_samples
        self.entropy_reg = entropy_reg
        self.reward_transformer = reward_transformer
        self.termination = termination

    def forward(self, states):
        """Compute the losses for one step of MPO.

        Note to future self: MPPO uses the reversed mode-seeking KL-divergence.
        Don't change the next direction of the KL divergence.

        TRPO/PPO use the forward mean-seeking KL-divergence.
        kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)

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

        pi_dist = tensor_to_distribution(self.policy(states, normalized=True))
        pi_dist_old = tensor_to_distribution(self.old_policy(states, normalized=True))
        kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)

        with torch.no_grad():
            value_estimate, trajectory = mb_return(
                state=states, dynamical_model=self.dynamical_model,
                policy=self.old_policy,
                reward_model=self.reward_model, num_steps=1, gamma=self.gamma,
                value_function=self.value_target,
                num_samples=self.num_action_samples,
                reward_transformer=self.reward_transformer,
                entropy_reg=self.entropy_reg, termination=self.termination)
        q_values = value_estimate
        action_log_probs = pi_dist.log_prob(
            trajectory[0].action / self.policy.action_scale)

        # Since actions come from policy, value is the expected q-value
        losses = self.mppo_loss(q_values=q_values, action_log_probs=action_log_probs,
                                kl_mean=kl_mean, kl_var=kl_var)

        value_loss = self.value_loss(value_prediction, q_values.mean(dim=0))
        td_error = value_prediction - q_values.mean(dim=0)

        dual_loss = losses.dual_loss.mean()
        policy_loss = losses.primal_loss.mean()
        combined_loss = value_loss + dual_loss + policy_loss

        self._info = {'kl_div': kl_mean + kl_var, 'kl_mean': kl_mean, 'kl_var': kl_var,
                      'eta': self.mppo_loss.eta(),
                      'eta_mean': self.mppo_loss.eta_mean(),
                      'eta_var': self.mppo_loss.eta_var()
                      }

        return MPOLoss(loss=combined_loss, dual=dual_loss, policy_loss=policy_loss,
                       critic_loss=value_loss, td_error=td_error)

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self):
        """Update target value function."""
        update_parameters(self.value_target, self.value_function,
                          tau=self.value_function.tau)
