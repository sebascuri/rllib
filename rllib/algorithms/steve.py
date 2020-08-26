"""Stochastic Ensemble Value Expansion Algorithm."""
import torch

from rllib.model.utilities import PredictionStrategy
from rllib.util.value_estimation import mc_return

from .abstract_mb_algorithm import AbstractMBAlgorithm


def steve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination=None,
    *args,
    **kwargs,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class STEVE(type(base_algorithm), AbstractMBAlgorithm):
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
            AbstractMBAlgorithm.__init__(
                self,
                dynamical_model,
                reward_model,
                num_steps=num_steps,
                num_samples=num_samples,
                termination=termination,
            )

            try:
                num_q = self.critic_target.num_heads
            except AttributeError:
                num_q = 1

            self.num_models = self.dynamical_model.base_model.num_heads
            self.num_q = num_q

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            critic_target = torch.zeros(
                observation.state.shape[: -len(self.dynamical_model.dim_state)]
                + (self.num_steps + 1, self.num_models, self.num_q)
            )  # Critic target shape B x (H + 1) x M x Q

            td_target = super().get_value_target(observation)  # TD-Target B x 1.
            critic_target[..., 0, :, :] = (
                td_target.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.num_models, dim=-2)
                .repeat_interleave(self.num_q, dim=-1)
            )

            with PredictionStrategy(
                self.dynamical_model, prediction_strategy="set_head"
            ), torch.no_grad():
                state = observation.state[..., 0, :]

                for model_idx in range(self.num_models):  # Rollout each model.
                    self.dynamical_model.set_head(model_idx)
                    trajectory = self.simulate(state, self.policy)
                    for horizon in range(self.num_steps):  # Compute different targets.
                        value = mc_return(
                            trajectory[: (horizon + 1)],
                            gamma=self.gamma,
                            value_function=self.value_target,
                            reward_transformer=self.reward_transformer,
                        )
                        value = value.mean(0).unsqueeze(1)
                        critic_target[..., horizon + 1, model_idx, :] = value

            mean_target = critic_target.mean(dim=(-1, -2))
            weight_target = 1 / (self.eps + critic_target.var(dim=(-1, -2)))

            weights = weight_target / weight_target.sum(-1, keepdim=True)
            critic_target = (weights * mean_target).sum(-1)

            return critic_target

    return STEVE()
