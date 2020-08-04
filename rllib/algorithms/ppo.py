"""Proximal Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.util.neural_networks import resume_learning
from rllib.util.parameter_decay import Constant, ParameterDecay

from .abstract_algorithm import TRPOLoss
from .trpo import TRPO


class PPO(TRPO):
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

    def __init__(
        self,
        epsilon=0.2,
        weight_value_function=1.0,
        weight_entropy=0.01,
        clamp_value=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(epsilon, ParameterDecay):
            epsilon = Constant(epsilon)
        self.epsilon = epsilon

        self.weight_value_function = weight_value_function
        self.weight_entropy = weight_entropy

        self.clamp_value = clamp_value

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        super().reset()
        # Resume learning after early stopping.
        resume_learning(self.policy)

    def forward_slow(self, trajectories):
        """Compute losses slowly, iterating through the trajectories."""
        surrogate_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        kl_div = torch.tensor(0.0)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory
            with torch.no_grad():
                adv = self.returns(trajectory)
                if self.monte_carlo_target:
                    value_target = self.get_q_target(trajectory)
                else:
                    value_target = adv + self.value_function_target(state)

            log_p, log_p_old, kl_mean_, kl_var_, entropy = self.get_log_p_kl_entropy(
                state, action
            )
            kl_div_ = (kl_mean_ + kl_var_).mean()

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

        return TRPOLoss(
            loss=combined_loss / num_trajectories,
            critic_loss=value_loss / num_trajectories,
            surrogate_loss=surrogate_loss / num_trajectories,
            kl_loss=entropy_bonus / num_trajectories,
            dual_loss=torch.tensor(0.0),
        )
