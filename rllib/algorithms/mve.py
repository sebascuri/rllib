"""Model Based Value Expansion Algorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.value_estimation import discount_cumsum, mc_return
from rllib.value_function import (
    AbstractQFunction,
    AbstractValueFunction,
    IntegrateQValueFunction,
)


def mve_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination=None,
    td_k=True,
):
    """Expand a MVE-Expanded algorithm.

    Given an algorithm, return the target via simulation of a model.
    """
    #

    class MVE(type(base_algorithm)):
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

            self.criterion = type(self.criterion)(reduction="mean")
            self.td_k = td_k

        def simulate(self, observation):
            """Simulate starting from an observation."""
            state = repeat_along_dimension(
                observation.state[..., 0, :], number=self.num_samples, dim=0
            )
            trajectory = rollout_model(
                self.dynamical_model,
                self.reward_model,
                self.policy,
                state,
                max_steps=self.num_steps,
                termination=self.termination,
            )

            return trajectory

        def critic_loss(self, observation):
            """Get critic-loss by rolling out a model."""
            if self.td_k:
                with torch.no_grad():
                    trajectory = self.simulate(observation)
                observation = stack_list_of_tuples(trajectory, dim=2)

            return super().critic_loss(observation)

        def get_value_target(self, observation):
            """Rollout model and call base algorithm with transitions."""
            if self.td_k:
                assert observation.reward.shape[-1] == self.num_steps
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
                    trajectory = self.simulate(observation)
                    target_q = (
                        mc_return(
                            trajectory,
                            gamma=self.gamma,
                            value_function=self.value_target,
                            reward_transformer=self.reward_transformer,
                        )
                        .unsqueeze(-2)
                        .mean(0)
                    )

            return target_q

    return MVE()
