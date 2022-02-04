"""ModelBasedAlgorithm."""
import torch

from .abstract_mb_algorithm import AbstractMBAlgorithm
from .derived_algorithm import DerivedAlgorithm


class Dyna(DerivedAlgorithm, AbstractMBAlgorithm):
    """Dyna Algorithm."""

    def __init__(
        self,
        base_algorithm,
        dynamical_model,
        reward_model,
        num_model_steps=1,
        num_particles=15,
        termination_model=None,
        only_sim=False,
        only_real=False,
        *args,
        **kwargs,
    ):
        DerivedAlgorithm.__init__(self, base_algorithm=base_algorithm)
        AbstractMBAlgorithm.__init__(
            self,
            dynamical_model,
            reward_model,
            num_model_steps=num_model_steps,
            num_particles=num_particles,
            termination_model=termination_model,
        )
        self.base_algorithm.criterion = type(self.base_algorithm.criterion)(
            reduction="mean"
        )
        self.only_sim = only_sim
        self.only_real = only_real
        assert not only_sim or not only_real, "only one can be True."

    def forward(self, observation):
        """Rollout model and call base algorithm with transitions."""
        real_loss = self.base_algorithm.forward(observation)
        with torch.no_grad():
            state = observation.state[..., 0, :]
            sim_observation = self.simulate(state, self.policy, stack_obs=True)
        sim_loss = self.base_algorithm.forward(sim_observation)
        if self.only_sim:
            return sim_loss
        if self.only_real:
            return real_loss
        return real_loss + sim_loss
