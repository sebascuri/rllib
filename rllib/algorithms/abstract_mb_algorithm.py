"""ModelBasedAlgorithm."""
from abc import ABCMeta

import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import TransformedModel

from .abstract_algorithm import AbstractAlgorithm
from .simulation_algorithm import SimulationAlgorithm


class AbstractMBAlgorithm(AbstractAlgorithm, metaclass=ABCMeta):
    """Model Based Algorithm.

    A model based algorithm has a dynamical_model and a reward_model and, it has a
    simulate method that simulates trajectories following a policy.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        Dynamical model to simulate.
    reward_model: AbstractReward.
        Reward model to simulate.
    num_model_steps: int.
        Number of steps to simulate.
    num_particles: int.
        Number of parallel samples to simulate.
    termination: Termination, optional.
        Termination condition to evaluate while simulating.

    Methods
    -------
    simulate(self, state: State, policy: AbstractPolicy) -> Trajectory:
        Simulate a set of particles starting from `state' and following `policy'.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_model_steps=1,
        num_particles=1,
        termination_model=None,
        log_simulation=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        if not isinstance(reward_model, TransformedModel):
            reward_model = TransformedModel(reward_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model

        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.log_simulation = log_simulation
        self.num_model_steps = num_model_steps
        self.num_particles = num_particles

        self.simulation_algorithm = SimulationAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            num_particles=num_particles,
            num_model_steps=num_model_steps,
        )

    def simulate(
        self, initial_state, policy, initial_action=None, stack_obs=True, memory=None
    ):
        """Simulate a set of particles starting from `state' and following `policy'."""
        trajectory = self.simulate(
            state=initial_state,
            policy=policy,
            initial_action=initial_action,
            memory=memory,
        )
        if not stack_obs:
            self._log_trajectory(trajectory)
            return trajectory
        else:
            observation = stack_list_of_tuples(trajectory, dim=initial_state.ndim - 1)
            self._log_observation(observation)
            return observation

    def _log_trajectory(self, trajectory):
        """Log the simulated trajectory."""
        observation = stack_list_of_tuples(trajectory, dim=trajectory[0].state.ndim - 1)
        self._log_observation(observation)

    def _log_observation(self, observation):
        """Log a simulated observation (a stacked trajectory)."""
        if not self.log_simulation:
            return
        scale = torch.diagonal(observation.next_state_scale_tril, dim1=-1, dim2=-2)
        self._info.update(
            sim_entropy=observation.entropy.mean().item(),
            sim_return=observation.reward.sum(-1).mean().item(),
            sim_scale=scale.square().sum(-1).sum(0).mean().sqrt().item(),
            sim_max_state=observation.state.abs().max().item(),
            sim_max_action=observation.action.abs().max().item(),
        )
        for key, value in self.reward_model.info.items():
            self._info.update(**{f"sim_{key}": value})
