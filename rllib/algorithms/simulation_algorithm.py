"""Python Script Template."""
import gpytorch
import torch

from rllib.dataset.experience_replay import StateExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import DisableGradient

from .abstract_mb_algorithm import AbstractMBAlgorithm


class SimulationAlgorithm(AbstractMBAlgorithm):
    """Base class for simulation algorithms."""

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
        super().__init__(*args, **kwargs)
        self.initial_distribution = initial_distribution

        self.num_initial_state_samples = num_initial_state_samples
        self.num_initial_distribution_samples = num_initial_distribution_samples
        self.num_memory_samples = num_memory_samples

        self.num_subsample = num_subsample
        self.refresh_interval = refresh_interval
        self._idx = 0

        self.dataset = StateExperienceReplay(
            max_len=max_memory, dim_state=self.dynamical_model.dim_state
        )

    def get_initial_states(self, initial_states_dataset, real_dataset):
        """Get initial states to sample from."""
        # Samples from empirical initial state distribution.
        initial_states = initial_states_dataset.get_batch(
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
            obs, *_ = real_dataset.sample_batch(real_dataset)
            for transform in real_dataset.transformations:
                obs = transform.inverse(obs)
            initial_states_ = obs.state[:, 0, :]  # obs is an n-step return.
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)
        return initial_states

    def simulate(self, state, policy):
        """Simulate from initial_states."""
        self.dynamical_model.eval()

        with DisableGradient(
            self.dynamical_model, self.reward_model
        ), gpytorch.settings.fast_pred_var():
            trajectory = super().simulate(state, policy)

        sim_trajectory = stack_list_of_tuples(trajectory)
        states = sim_trajectory.state.reshape(-1, *self.dynamical_model.dim_state)
        self.dataset.append(states[:: self.num_subsample])

        if self.refresh_interval > 0 and (self._idx + 1) % self.refresh_interval == 0:
            self.dataset.reset()
        self._idx += 1
        return trajectory
