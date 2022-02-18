"""ModelBasedAlgorithm."""
from abc import ABCMeta

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
