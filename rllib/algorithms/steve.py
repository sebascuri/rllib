"""Stochastic Ensemble Value Expansion Algorithm."""
import torch

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.value_estimation import mc_return
from rllib.value_function import (
    AbstractQFunction,
    AbstractValueFunction,
    IntegrateQValueFunction,
)


def steve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination=None,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class STEVE(type(base_algorithm)):
        """Stochastic Ensemble Algorithm using STEVE to calculate targets.

        Overrides get_value_target() method.

        References
        ----------
        Buckman, J., Hafner, D., Tucker, G., Brevdo, E., & Lee, H. (2018).
        Sample-efficient reinforcement learning with stochastic ensemble value
        expansion. NeuRIPS.
        """

        def __init__(self):
            super().__init__(
                **{**base_algorithm.__dict__, **dict(base_algorithm.named_modules())}
            )
            self.dynamical_model = dynamical_model

            self.reward_model = reward_model
            self.num_steps = num_steps
            self.num_samples = num_samples
            self.termination = termination

            if isinstance(self.critic_target, AbstractQFunction):
                self.value_target = IntegrateQValueFunction(
                    self.critic_target, self.policy, num_samples=self.num_samples
                )
            elif isinstance(self.critic_target, AbstractValueFunction):
                self.value_target = self.critic_target
            else:
                self.value_target = None

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            num_models = self.dynamical_model.base_model.num_heads

            if isinstance(self.value_target, IntegrateQValueFunction):
                num_q = self.value_target.q_function.num_heads
            else:
                num_q = self.value_target.num_heads

            critic_target = torch.zeros(
                observation.state.shape[: -len(self.dynamical_model.dim_state)]
                + (self.num_steps + 1, num_models, num_q)
            )  # Critic target shape B x (H + 1) x M x Q

            td_target = super().get_value_target(observation)  # TD-Target B x 1.
            critic_target[..., 0, :, :] = (
                td_target.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(num_models, dim=-2)
                .repeat_interleave(num_q, dim=-1)
            )

            current_head = self.dynamical_model.base_model.get_head()
            current_pred = self.dynamical_model.base_model.get_prediction_strategy()
            self.dynamical_model.base_model.set_prediction_strategy("set_head")

            with torch.no_grad():
                state = repeat_along_dimension(
                    observation.state[..., 0, :], number=self.num_samples, dim=0
                )

                for model_idx in range(num_models):  # Rollout each model.
                    self.dynamical_model.base_model.set_head(model_idx)
                    trajectory = rollout_model(
                        self.dynamical_model,
                        self.reward_model,
                        self.policy,
                        state,
                        max_steps=self.num_steps,
                        termination=self.termination,
                    )
                    for horizon in range(self.num_steps):  # Compute different targets.
                        value = (
                            mc_return(
                                trajectory[: (horizon + 1)],
                                gamma=self.gamma,
                                value_function=self.value_target,
                                reward_transformer=self.reward_transformer,
                            )
                            .mean(0)
                            .unsqueeze(1)
                        )

                        critic_target[..., horizon + 1, model_idx, :] = value

            mean_target = critic_target.mean(dim=(-1, -2))
            weight_target = 1 / (self.eps + critic_target.var(dim=(-1, -2)))

            weights = weight_target / weight_target.sum(-1, keepdim=True)

            critic_target = (weights * mean_target).sum(-1)

            self.dynamical_model.base_model.set_head(current_head)
            self.dynamical_model.base_model.set_prediction_strategy(current_pred)

            return critic_target

    return STEVE()
