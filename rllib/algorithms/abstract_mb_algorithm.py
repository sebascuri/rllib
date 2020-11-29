"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import TransformedModel
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model


class AbstractMBAlgorithm(object):
    """Model Based Algorithm.

    A model based algorithm has a dynamical_model and a reward_model and, it has a
    simulate method that simulates trajectories following a policy.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        Dynamical model to simulate.
    reward_model: AbstractReward.
        Reward model to simulate.
    num_steps: int.
        Number of steps to simulate.
    num_samples: int.
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
        num_steps=1,
        num_samples=1,
        termination_model=None,
        log_simulation=False,
        *args,
        **kwargs,
    ):
        super().__init__()
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
        self.num_steps = num_steps
        self.num_samples = num_samples  # type: int
        self._info = dict()  # type: dict

    def simulate(
        self, initial_state, policy, initial_action=None, logger=None, stack_obs=True
    ):
        """Simulate a set of particles starting from `state' and following `policy'."""
        if self.num_samples > 0:
            initial_state = repeat_along_dimension(
                initial_state, number=self.num_samples, dim=0
            )
            initial_state = initial_state.reshape(-1, *self.dynamical_model.dim_state)
            if initial_action is not None:
                initial_action = repeat_along_dimension(
                    initial_action, number=self.num_samples, dim=0
                )
                initial_action = initial_action.reshape(*initial_state.shape[:-1], -1)

        trajectory = rollout_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=policy,
            initial_state=initial_state,
            initial_action=initial_action,
            max_steps=self.num_steps,
            termination_model=self.termination_model,
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
