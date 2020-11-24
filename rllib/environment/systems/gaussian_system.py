"""Implementation of a System with Gaussian transition and measurement noise."""

import numpy as np

from .abstract_system import AbstractSystem


class GaussianNoiseSystem(AbstractSystem):
    """Modify a system with gaussian transition and measurement noise.

    Parameters
    ----------
    system: AbstractSystem
    transition_noise_scale: float
    measurement_noise_scale: float, optional

    """

    def __init__(self, system, transition_noise_scale, measurement_noise_scale=0):
        super().__init__(
            dim_state=system.dim_state,
            dim_action=system.dim_action,
            dim_observation=system.dim_observation,
        )
        self._system = system
        self._transition_noise_scale = transition_noise_scale
        self._measurement_noise_scale = measurement_noise_scale

    def step(self, action):
        """See `AbstractSystem.step'."""
        next_state = self._system.step(action)
        next_state += self._transition_noise_scale * np.random.randn(self.dim_state)
        return next_state

    def reset(self, state):
        """See `AbstractSystem.reset'."""
        return self._system.reset(state)

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        state = self._system.state
        state += self._measurement_noise_scale * np.random.randn(self.dim_state)
        return state

    @state.setter
    def state(self, value):
        self._system.state = value
