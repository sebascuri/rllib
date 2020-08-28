"""Model Based Value Expansion Algorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.value_estimation import discount_cumsum, mc_return

from .abstract_mb_algorithm import AbstractMBAlgorithm


def mve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination=None,
    td_k=True,
    *args,
    **kwargs,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class MVE(type(base_algorithm), AbstractMBAlgorithm):
        """Derived Algorithm using MVE to calculate targets.

        References
        ----------
        Feinberg, V., et. al. (2018).
        Model-based value estimation for efficient model-free reinforcement learning.
        arXiv.
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

            self.criterion = type(self.criterion)(reduction="mean")
            self.td_k = td_k

        def critic_loss(self, observation):
            """Get critic-loss by rolling out a model."""
            if self.td_k:
                with torch.no_grad():
                    state = observation.state[..., 0, :]
                    trajectory = self.simulate(state, self.policy)
                observation = stack_list_of_tuples(trajectory, dim=-2)

            return super().critic_loss(observation)

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            if self.td_k:
                if self.critic.discrete_state:
                    final_state = observation.next_state[..., -1]
                else:
                    final_state = observation.next_state[..., -1, :]
                done = observation.done[..., -1]
                final_value = self.value_target(final_state)

                if final_value.ndim == observation.reward.ndim:  # It is an ensemble.
                    final_value = final_value.min(-1)[0]

                rewards = torch.cat(
                    (observation.reward, (final_value * (1 - done)).unsqueeze(-1)),
                    dim=-1,
                )
                target_q = discount_cumsum(
                    rewards, self.gamma, self.reward_transformer
                )[..., :-1]

            else:
                with torch.no_grad():
                    state = observation.state[..., 0, :]
                    trajectory = self.simulate(state, self.policy)
                    target_q = mc_return(
                        trajectory,
                        gamma=self.gamma,
                        value_function=self.value_target,
                        reward_transformer=self.reward_transformer,
                    )
                    target_q = (
                        target_q.reshape(
                            self.num_samples, *observation.state.shape[:-1], -1
                        )
                        .mean(0)
                        .squeeze(-1)
                    )

            return target_q

    return MVE()
