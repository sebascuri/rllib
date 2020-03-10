"""Underwater Vehicle Implementation."""

from rllib.environment.systems.ode_system import ODESystem
from rllib.environment.systems.linear_system import LinearSystem
# from scipy import signal
import numpy as np


class UnderwaterVehicle(ODESystem):
    """Underwater Vehicle.

    Parameters
    ----------
    step_size : float, optional

    References
    ----------
    Hafner, R., & Riedmiller, M. (2011).
     Reinforcement learning in feedback control. Machine learning.
    """

    def __init__(self, step_size=0.01):
        super().__init__(
            func=self._ode,
            step_size=step_size,
            dim_action=1,
            dim_state=1)

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        ad : ndarray
            The discrete-time state matrix.
        bd : ndarray
            The discrete-time action matrix.

        """
        a = np.zeros(self.dim_state, self.dim_state)
        b = np.ones(self.dim_state, self.dim_action)
        return LinearSystem(a, b)

    def drag(self, velocity):
        """Get drag coefficient."""
        return 1.2 + 0.2 * np.sin(np.abs(velocity))

    def mass(self, velocity):
        """Get mass coefficient."""
        return 3.0 + 1.5 * np.sin(np.abs(velocity))

    def thrust(self, velocity, thrust):
        """Get the thrust coefficient."""
        drag_force = self.drag(velocity) * velocity * np.abs(velocity)
        return -0.5 * np.tanh(0.1 * (np.abs(drag_force - thrust) - 30.0)) + 0.5

    def _ode(self, _, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.
        action: ndarray or Tensor
            Actions.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        v = state
        u = action
        c = self.drag(v)
        m = self.mass(v)
        k = self.thrust(v, u)
        v_dot = (k * u - c * v * np.abs(v)) / m

        return v_dot
