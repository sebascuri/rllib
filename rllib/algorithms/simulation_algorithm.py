"""Python Script Template."""
import gpytorch
import torch

from rllib.util.neural_networks.utilities import DisableGradient

from .abstract_mb_algorithm import AbstractMBAlgorithm


class SimulationAlgorithm(AbstractMBAlgorithm):
    """An algorithm for simulating trajectories and storing them in a data set.

    Parameters
    ----------
    initial_distribution: Distribution, optional.
        Initial distribution to sample from.
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
        num_initial_state_samples=0,
        num_initial_distribution_samples=0,
        num_memory_samples=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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

    def get_initial_states(self, initial_states_dataset, real_dataset):
        """Get initial states to sample from."""
        # Samples from empirical initial state distribution.
        initial_states = initial_states_dataset.sample_batch(
            max(self.num_initial_state_samples, 1)
            # hack: f
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

    def simulate(
        self, state, policy, initial_action=None, logger=None, stack_obs=False
    ):
        """Simulate from initial_states."""
        self.dynamical_model.eval()
        with DisableGradient(
            self.dynamical_model, self.reward_model, self.termination_model
        ), gpytorch.settings.fast_pred_var():
            trajectory = super().simulate(state, policy, stack_obs=stack_obs)

        self._log_trajectory(trajectory)
        return trajectory
