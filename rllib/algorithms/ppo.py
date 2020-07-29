"""Proximal Policy Optimization algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.algorithms.gae import GAE
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    unfreeze_parameters,
    update_parameters,
)
from rllib.util.parameter_decay import Constant, ParameterDecay
from rllib.util.utilities import tensor_to_distribution
from rllib.util.value_estimation import discount_cumsum

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
    criterion: Type[_Loss], optional.
        Criterion for value function.
    epsilon: float, optional (default=0.2)
        The clipping parameter.
    weight_value_function: float, optional. (default=1).
        Weight of the value function loss relative to the surrogate loss.
    weight_entropy: float, optional. (default=0).
        Weight of the entropy bonus relative to the surrogate objective.
    monte_carlo_target: bool, optional. (default=False).
        Whether to calculate the value targets using MC estimation or adv + value.
    clamp_value: bool, optional. (default=False).
        Whether to clamp the value estimate before computing the value loss.
    lambda_: float, optional. (default=0.97).
        Parameter for Advantage estimation.
    gamma: float, optional. (default=1).
        Discount factor.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms. ArXiv.

    Engstrom, L., et al. (2020).
    Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR.
    """

    eps = 1e-5

    def __init__(
        self,
        policy,
        value_function,
        criterion=nn.MSELoss,
        epsilon=0.2,
        weight_value_function=1.0,
        weight_entropy=0.01,
        monte_carlo_target=False,
        clamp_value=False,
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

        if not isinstance(epsilon, ParameterDecay):
            epsilon = Constant(epsilon)
        self.epsilon = epsilon

        self.value_loss_criterion = criterion(reduction="mean")
        self.weight_value_function = weight_value_function
        self.weight_entropy = weight_entropy

        self.gae = GAE(
            lambda_=lambda_, gamma=self.gamma, value_function=self.value_function_target
        )
        self.monte_carlo_target = monte_carlo_target
        self.clamp_value = clamp_value

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())
        unfreeze_parameters(self.policy)

    def get_log_p_and_entropy(self, state, action):
        """Get log probability, entropy and kl_divergence at sampled state-actions.

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
        entropy: torch.Tensor
            Entropy of current policy.
        kl_div: torch.Tensor
            KL-Divergence divergence between current and previous policy.
        """
        pi_dist = tensor_to_distribution(self.policy(state))
        pi_dist_old = tensor_to_distribution(self.old_policy(state))

        log_p, log_p_old = pi_dist.log_prob(action), pi_dist_old.log_prob(action)

        try:
            kl_div = torch.distributions.kl_divergence(p=pi_dist_old, q=pi_dist).mean()
            entropy = pi_dist.entropy().mean()
        except NotImplementedError:
            kl_div = (log_p_old - log_p).mean()
            entropy = -log_p.mean()

        return log_p, log_p_old, entropy, kl_div

    def get_advantage_and_value_target(self, trajectory):
        """Get advantage and value targets."""
        state, action, reward, next_state, done, *r = trajectory

        adv = self.gae(trajectory)
        adv = (adv - adv.mean()) / (adv.std() + self.eps)

        if self.monte_carlo_target:
            final_state = next_state[-1:]
            reward_to_go = self.value_function_target(final_state) * (1.0 - done[-1:])
            value_target = discount_cumsum(
                torch.cat((reward, reward_to_go)), gamma=self.gamma
            )[:-1]
            value_target = (value_target - value_target.mean()) / (
                value_target.std() + self.eps
            )
        else:
            value_target = adv + self.value_function_target(state)

        return adv, value_target

    def forward_slow(self, trajectories):
        """Compute losses slowly, iterating through the trajectories."""
        surrogate_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        kl_div = torch.tensor(0.0)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory
            with torch.no_grad():
                adv, value_target = self.get_advantage_and_value_target(trajectory)

            log_p, log_p_old, entropy, kl_div_ = self.get_log_p_and_entropy(
                state, action
            )

            # Compute Value loss.
            value_pred = self.value_function(state)
            if self.clamp_value:
                old_value_pred = self.value_function_target(state).detach()
                value_pred = torch.max(
                    torch.min(value_pred, old_value_pred + self.epsilon()),
                    old_value_pred - self.epsilon(),
                )

            value_loss += self.value_loss_criterion(value_pred, value_target)

            # Compute surrogate loss.
            ratio = torch.exp(log_p - log_p_old)
            clip_adv = ratio.clamp(1 - self.epsilon(), 1 + self.epsilon()) * adv
            surrogate_loss -= torch.min(ratio * adv, clip_adv).mean()

            # Compute entropy bonus.
            entropy_bonus += entropy

            # Compute exact and approximate KL divergence
            kl_div += kl_div_

        num_trajectories = len(trajectories)
        combined_loss = (
            surrogate_loss
            + self.weight_value_function * value_loss
            - self.weight_entropy * entropy_bonus
        )

        self._info = {"kl_div": kl_div / num_trajectories}

        return PPOLoss(
            loss=combined_loss / num_trajectories,
            critic_loss=value_loss / num_trajectories,
            surrogate_loss=surrogate_loss / num_trajectories,
            entropy=entropy_bonus / num_trajectories,
        )

    def forward(self, trajectories):
        """Compute the losses a trajectory.

        Parameters
        ----------
        trajectories: List[torch.Tensor]

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
        if len(trajectories) > 1:
            try:  # When possible, paralelize the trajectories.
                trajectories = [stack_list_of_tuples(trajectories)]
            except RuntimeError:
                pass
        return self.forward_slow(trajectories)

    def update(self):
        """Update the baseline network."""
        update_parameters(
            self.value_function_target, self.value_function, self.value_function.tau
        )
