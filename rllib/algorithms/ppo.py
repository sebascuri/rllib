"""Proximal Policy Optimization algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.algorithms.gae import GAE
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    update_parameters,
)
from rllib.util.parameter_decay import Constant, ParameterDecay
from rllib.util.utilities import tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm, PPOLoss


class PPO(AbstractAlgorithm):
    """Proximal Policy Optimization algorithm..

    The PPO algorithm returns a loss that is a combination of three losses.

    - The clipped surrogate objective (Eq. 7).
    - The value function error (Eq. 9).
    - A policy entropy bonus.

    Parameters
    ----------
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    epsilon: float, optional (default=0.2)
        The clipping parameter.
    weight_value_function: float, optional. (default=1).
        Weight of the value function loss relative to the surrogate loss.
    weight_entropy: float, optional. (default=0).
        Weight of the entropy bonus relative to the surrogate objective.
    lambda_: float, optional. (default=0.97).
        Parameter for Advantage estimation.
    gamma: float, optional. (default=1).
        Discount factor.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms. ArXiv.
    """

    def __init__(
        self,
        policy,
        value_function,
        epsilon=0.2,
        weight_value_function=1,
        weight_entropy=0,
        lambda_=0.97,
        gamma=1.0,
    ):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.policy = policy
        self.value_function = value_function
        self.value_function_target = deep_copy_module(value_function)

        self.gamma = gamma

        if not isinstance(epsilon, ParameterDecay):
            epsilon = Constant(epsilon)
        self.epsilon = epsilon

        self.value_loss = nn.MSELoss(reduction="mean")
        self.weight_value_function = weight_value_function
        self.weight_entropy = weight_entropy

        self.gae = GAE(lambda_, self.gamma, self.value_function_target)

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

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
        entropy_bonus = torch.tensor(0.0)
        kl_div = torch.tensor(0.0)
        approx_kl_div = torch.tensor(0.0)

        for trajectory in trajectories:
            with torch.no_grad():
                adv = self.gae(trajectory)

            state, action, reward, next_state, done, *r = trajectory
            pi = tensor_to_distribution(self.policy(state))
            pi_old = tensor_to_distribution(self.old_policy(state))
            log_p, log_p_old = pi.log_prob(action), pi_old.log_prob(action)

            # Compute Value loss.
            value_pred = self.value_function(state)
            value_loss += self.value_loss(value_pred, adv + value_pred.detach())

            # Compute surrogate loss.
            ratio = torch.exp(log_p - log_p_old)
            clip_adv = ratio.clamp(1 - self.clip_ratio(), 1 + self.clip_ratio()) * adv
            surrogate_loss -= torch.min(ratio * adv, clip_adv).mean()

            # Compute entropy bonus.
            entropy_bonus += pi.entropy()

            # Compute exact and approximate KL divergence
            kl_div += torch.distributions.kl_divergence(p=pi_old, q=pi).mean()
            approx_kl_div += (log_p_old - log_p).mean()

        combined_loss = (
            surrogate_loss
            + self.weight_value_function * value_loss
            + self.weight_entropy * entropy_bonus
        )

        self._info = {"kl_div": kl_div, "approx_kl_div": approx_kl_div}

        return PPOLoss(
            loss=combined_loss,
            critic_loss=value_loss,
            surrogate_loss=surrogate_loss,
            entropy=entropy_bonus,
        )

    def update(self):
        """Update the baseline network."""
        update_parameters(
            self.value_function_target, self.value_function, self.value_function.tau
        )
