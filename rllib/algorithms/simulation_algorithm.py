"""Simulation algorithm."""

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model


class SimulationAlgorithm(object):
    """An algorithm for simulating trajectories and storing them in a data set.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        Model of the dynamics.
    reward_model: AbstractModel.
        Model of the Rewards.
    termination_model: AbstractModel.
        Termination Model.
    num_particles: int.
        Number of particles to simulate from initial state.
    num_model_steps: int.
        Number of steps to simulate the particles. .

    Methods
    -------
    get_initial_states(
        self,
        initial_states_dataset: StateExperienceReplay,
        real_dataset: ExperienceReplay
    ) -> Tensor:
        Get initial states for simulation.
    simulate(self, state: State, policy: AbstractPolicy) -> Trajectory:
        Simulate a set of particles starting from `state' and following `policy'.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        termination_model=None,
        num_particles=1,
        num_model_steps=1,
    ):
        super().__init__()
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        self.num_particles = num_particles
        self.num_model_steps = num_model_steps

    def simulate(self, initial_state, policy, initial_action=None, memory=None):
        """Simulate a set of particles starting from `state' and following `policy'."""
        if self.num_particles > 0:
            initial_state = repeat_along_dimension(
                initial_state, number=self.num_particles, dim=0
            )
            initial_state = initial_state.reshape(-1, *self.dynamical_model.dim_state)
            if initial_action is not None:
                initial_action = repeat_along_dimension(
                    initial_action, number=self.num_particles, dim=0
                )
                initial_action = initial_action.reshape(*initial_state.shape[:-1], -1)

        trajectory = rollout_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=policy,
            initial_state=initial_state,
            initial_action=initial_action,
            max_steps=self.num_model_steps,
            termination_model=self.termination_model,
            memory=memory,
        )
        return trajectory
