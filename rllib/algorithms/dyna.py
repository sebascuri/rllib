"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples

from .abstract_mb_algorithm import AbstractMBAlgorithm


class Dyna(AbstractMBAlgorithm):
    """Dyna Algorithm."""

    def __init__(
        self, base_algorithm, only_sim=False, only_real=False, *args, **kwargs
    ):
        super().__init__(
            critic=base_algorithm.critic, policy=base_algorithm.policy, *args, **kwargs
        )
        self.base_algorithm = base_algorithm
        self.only_sim = only_sim
        self.only_real = only_real
        assert not only_sim or not only_real, "only one can be True."

    def forward(self, observation):
        """Rollout model and call base algorithm with transitions."""
        real_loss = self.base_algorithm(observation)
        if self.only_real:
            return real_loss

        with torch.no_grad():
            state = observation.state[..., 0, :]
            sim_trajectory = self.simulation_algorithm.simulate(state, self.policy)
            sim_observation = stack_list_of_tuples(sim_trajectory, dim=-2)

        sim_loss = self.base_algorithm(sim_observation)
        if self.only_sim:
            return sim_loss

        return real_loss.reduce("mean") + sim_loss.reduce("mean")

    def update(self):
        """Update base algorithm."""
        super().update()
        self.base_algorithm.update()

    def reset(self):
        """Reset base algorithm."""
        super().reset()
        self.base_algorithm.reset()

    def info(self):
        """Get info from base algorithm."""
        return self.base_algorithm.info()

    def reset_info(self):
        """Reset info from base algorithm."""
        super().reset_info()
        self.base_algorithm.reset_info()

    def set_policy(self, new_policy):
        """Set policy in base algorithm."""
        super().set_policy(new_policy)
        self.base_algorithm.set_policy(new_policy)
