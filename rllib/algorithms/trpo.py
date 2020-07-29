"""Trust Region Policy Optimization algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.algorithms.gae import GAE
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    update_parameters,
)
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import separated_kl, tensor_to_distribution
from rllib.util.value_estimation import discount_cumsum

from .abstract_algorithm import AbstractAlgorithm, TRPOLoss


class TRPO(AbstractAlgorithm):
    """Trust Region Policy Optimization algorithm..

    The TRPO algorithm returns a loss that is a combination of three losses.

    - The surrogate objective.
    - The value function error.
    - The dual loss that arises from the violation of the KL constraint.

    Parameters
    ----------
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    regularization: bool.
        Flag to indicate whether to regularize or do a trust region.
    epsilon_mean: ParameterDecay, float, optional.
        Kl Divergence of the Mean.
    epsilon_var: ParameterDecay, float, optional.
        Kl Divergence of the variance.
    lambda_: float, optional. (default=0.97).
        Parameter for Advantage estimation.
    gamma: float, optional. (default=1).
        Discount factor.

    References
    ----------
    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    Trust region policy optimization. ICML.
    """

    eps = 1e-5

    def __init__(
        self,
        policy,
        value_function,
        criterion=nn.MSELoss,
        regularization=False,
        epsilon_mean=0.2,
        epsilon_var=1e-4,
        lambda_=0.97,
        gamma=0.99,
    ):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.policy = policy
        self.value_function = value_function
        self.value_function_target = deep_copy_module(value_function)

        self.gamma = gamma

        if epsilon_var is None:
            epsilon_var = epsilon_mean

        if regularization:
            eta_mean = epsilon_mean
            eta_var = epsilon_var
            if not isinstance(eta_mean, ParameterDecay):
                eta_mean = Constant(eta_mean)
            if not isinstance(eta_var, ParameterDecay):
                eta_mean = Constant(eta_var)

            self.eta_mean = eta_mean
            self.eta_var = eta_var

            self.epsilon_mean = torch.tensor(0.0)
            self.epsilon_var = torch.tensor(0.0)

        else:  # Trust-Region: || KL(\pi_old || p) || < \epsilon
            self.eta_mean = Learnable(1.0, positive=True)
            self.eta_var = Learnable(1.0, positive=True)

            self.epsilon_mean = torch.tensor(epsilon_mean)
            self.epsilon_var = torch.tensor(epsilon_var)

        self.value_loss_criterion = criterion(reduction="mean")

        self.gae = GAE(
            lambda_=lambda_, gamma=self.gamma, value_function=self.value_function_target
        )

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_log_p_and_kl(self, state, action):
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

        action: torch.Tensor
            Actions sampled by pi_old.

        Returns
        -------
        log_p: torch.Tensor
            Log probability of actions according to current policy.
        log_p_old: torch.Tensor
            Log probability of actions according to old policy.
        kl_mean: torch.Tensor
            KL-Divergence due to the change in the mean between current and
            previous policy.
        kl_var: torch.Tensor
            KL-Divergence due to the change in the variance between current and
            previous policy.
        """
        pi_dist = tensor_to_distribution(self.policy(state))
        pi_dist_old = tensor_to_distribution(self.old_policy(state))

        log_p, log_p_old = pi_dist.log_prob(action), pi_dist_old.log_prob(action)

        if isinstance(pi_dist, torch.distributions.MultivariateNormal):
            kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)
        else:
            try:
                kl_mean = torch.distributions.kl_divergence(
                    p=pi_dist_old, q=pi_dist
                ).mean()
            except NotImplementedError:
                kl_mean = (log_p_old - log_p).mean()

            kl_var = torch.zeros_like(kl_mean)

        return log_p, log_p_old, kl_mean, kl_var

    def get_advantage_and_value_target(self, trajectory):
        """Get advantage and value targets."""
        state, action, reward, next_state, done, *r = trajectory

        adv = self.gae(trajectory)
        adv = (adv - adv.mean()) / (adv.std() + self.eps)

        # value_target = adv + self.value_function_target(state)
        final_state = next_state[-1:]
        reward_to_go = self.value_function_target(final_state) * (1.0 - done[-1:])
        value_target = discount_cumsum(
            torch.cat((reward, reward_to_go)), gamma=self.gamma
        )[:-1]
        value_target = (value_target - value_target.mean()) / (
            value_target.std() + self.eps
        )

        return adv, value_target

    def forward(self, trajectories):
        """Compute the losses a trajectory.

        Parameters
        ----------
        trajectories: torch.Tensor

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
        surrogate_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        dual_loss = torch.tensor(0.0)
        kl_loss = torch.tensor(0.0)

        kl_mean = torch.tensor(0.0)
        kl_var = torch.tensor(0.0)

        num_t = len(trajectories)
        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory
            value_pred = self.value_function(state)
            with torch.no_grad():
                adv, value_target = self.get_advantage_and_value_target(trajectory)

            log_p, log_p_old, kl_mean_, kl_var_ = self.get_log_p_and_kl(state, action)

            # Compute Value loss.
            value_loss += self.value_loss_criterion(value_pred, value_target)

            # Compute surrogate loss.
            ratio = torch.exp(log_p - log_p_old)
            surrogate_loss -= (ratio * adv).mean()

            # Compute KL Loss.
            kl_loss += (
                self.eta_mean().detach() * kl_mean_ + self.eta_var().detach() * kl_var_
            )

            # Compute Dual Loss.
            eta_mean_loss = self.eta_mean() * (self.epsilon_mean - kl_mean_.detach())
            eta_var_loss = self.eta_var() * (self.epsilon_var - kl_var_.detach())
            dual_loss += eta_mean_loss + eta_var_loss

            # Compute exact and approximate KL divergence
            kl_mean += kl_mean_
            kl_var += kl_var_

        combined_loss = surrogate_loss + kl_loss + dual_loss

        self._info = {
            "kl_div": (kl_mean + kl_var) / num_t,
            "kl_mean": kl_mean / num_t,
            "kl_var": kl_var / num_t,
            "eta_mean": self.eta_mean(),
            "eta_var": self.eta_var(),
        }

        return TRPOLoss(
            loss=combined_loss / num_t,
            critic_loss=value_loss / num_t,
            surrogate_loss=surrogate_loss / num_t,
            kl_loss=kl_loss,
            dual_loss=dual_loss / num_t,
        )

    def update(self):
        """Update the baseline network."""
        update_parameters(
            self.value_function_target, self.value_function, self.value_function.tau
        )
