"""Python Script Template."""
import gpytorch
import torch

from rllib.dataset.experience_replay import StateExperienceReplay
from rllib.util.neural_networks.utilities import DisableGradient

from .abstract_mb_algorithm import AbstractMBAlgorithm


class SimulationAlgorithm(AbstractMBAlgorithm):
    """An algorithm for simulating trajectories and storing them in a data set.

    Parameters
    ----------
    initial_distribution: Distribution, optional.
        Initial distribution to sample from.
    max_memory: int.
        Maximum size of simulation dataset.
    num_subsample: int.
        Subsample frequency of simulation data.
        If num_subsample=x, once every x simulated transitions will go into the dataset.
    num_initial_state_samples: int.
        Number of initial samples drawn from the initial_state dataset.
    num_initial_distribution_samples: int.
        Number of initial samples drawn from the initial distribution.
    num_memory_samples: int.
        Number of initial samples drawn from the full real dataset.
    refresh_interval: int.
        How often to erase the simulation dataset.

    Other Parameters
    ----------------
    See AbstractMBAlgorithm.

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
        initial_distribution=None,
        max_memory=100000,
        num_subsample=2,
        num_initial_state_samples=0,
        num_initial_distribution_samples=0,
        num_memory_samples=0,
        refresh_interval=2,
        *args,
        **kwargs,
    ):
        super().__init__(num_samples=0, *args, **kwargs)
        self.initial_distribution = initial_distribution

        self.num_initial_state_samples = num_initial_state_samples
        self.num_initial_distribution_samples = num_initial_distribution_samples
        self.num_memory_samples = num_memory_samples

        if self.num_initial_distribution_samples > 0:
            assert self.initial_distribution is not None
        assert (
            self.num_initial_state_samples
            + self.num_initial_distribution_samples
            + self.num_memory_samples
            > 0
        )

        self.num_subsample = num_subsample
        self.refresh_interval = refresh_interval
        self._idx = 0

        self.dataset = StateExperienceReplay(
            max_len=max_memory, dim_state=self.dynamical_model.dim_state
        )

    def get_initial_states(self, initial_states_dataset, real_dataset):
        """Get initial states to sample from."""
        # Samples from empirical initial state distribution.
        initial_states = initial_states_dataset.sample_batch(
            self.num_initial_state_samples
        )

        # Samples from initial distribution.
        if self.num_initial_distribution_samples > 0:
            initial_states_ = self.initial_distribution.sample(
                (self.num_initial_distribution_samples,)
            )
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        # Samples from experience replay empirical distribution.
        if self.num_memory_samples > 0:
            obs, *_ = real_dataset.sample_batch(self.num_memory_samples)
            for transform in real_dataset.transformations:
                obs = transform.inverse(obs)
            initial_states_ = obs.state[:, 0, :]  # obs is an n-step return.
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)
        return initial_states

    def simulate(self, state, policy, initial_action=None, logger=None):
        """Simulate from initial_states."""
        if self.refresh_interval > 0 and (self._idx + 1) % self.refresh_interval == 0:
            self.dataset.reset()
        self._idx += 1

        self.dynamical_model.eval()
        with DisableGradient(
            self.dynamical_model, self.reward_model, self.termination_model
        ), gpytorch.settings.fast_pred_var():
            observation = super().simulate(state, policy)

        states = observation.state.reshape(-1, *self.dynamical_model.dim_state)
        self.dataset.append(states[:: self.num_subsample])
        self._log_trajectory(observation, logger)
        return observation

    def _log_trajectory(self, stacked_trajectory, logger):
        """Log the simulated trajectory."""
        if logger is None:
            return
        scale = torch.diagonal(
            stacked_trajectory.next_state_scale_tril, dim1=-1, dim2=-2
        )
        logger.update(
            sim_entropy=stacked_trajectory.entropy.mean().item(),
            sim_return=stacked_trajectory.reward.sum(-1).mean().item(),
            sim_scale=scale.square().sum(-1).sum(0).mean().sqrt().item(),
            sim_max_state=stacked_trajectory.state.abs().max().item(),
            sim_max_action=stacked_trajectory.action.abs().max().item(),
        )
        for key, value in self.reward_model.info.items():
            logger.update(**{f"sim_{key}": value})
