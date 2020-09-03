"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples

from .abstract_mb_algorithm import AbstractMBAlgorithm


def dyna_expand(
    base_algorithm,
    dynamical_model,
    reward_model,
    num_steps=1,
    num_samples=15,
    termination_model=None,
    *args,
    **kwargs,
):
    """Expand algorithm with dyna simulations."""
    #

    class Dyna(type(base_algorithm), AbstractMBAlgorithm):
        """Model Based Algorithm.

        A model based algorithm simulates trajectories with a model.
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
                termination_model=termination_model,
            )
            self.policy.dist_params.update(**base_algorithm.policy.dist_params)
            self.policy_target.dist_params.update(
                **base_algorithm.policy_target.dist_params
            )

        def forward(self, observation, **kwargs_):
            """Rollout model and call base algorithm with transitions."""
            with torch.no_grad():
                state = observation.state[..., 0, :]
                trajectory = self.simulate(state, self.policy)
            try:
                observation = stack_list_of_tuples(trajectory, dim=-2)
                return super().forward(observation)
            except RuntimeError:
                return super().forward(trajectory)

    return Dyna()
