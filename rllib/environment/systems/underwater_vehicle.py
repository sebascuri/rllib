"""Underwater Vehicle Implementation."""

import numpy as np

from rllib.environment.systems.ode_system import ODESystem
from rllib.util.utilities import get_backend


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
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(1,)
        )

    def thrust(self, velocity, thrust):
        """Get the thrust coefficient."""
        bk = get_backend(velocity)
        return (
            -0.5 * bk.tanh(0.1 * (bk.abs(self.drag_force(velocity) - thrust) - 30.0))
            + 0.5
        )

    def drag_force(self, velocity):
        """Get drag force."""
        bk = get_backend(velocity)
        c = 1.2 + 0.2 * bk.sin(bk.abs(velocity))
        return c * velocity * bk.abs(velocity)

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
        velocity = state
        u = action

        bk = get_backend(velocity)
        m = 3.0 + 1.5 * bk.sin(bk.abs(velocity))  # mass
        k = self.thrust(velocity, u)  # thrust coefficient.
        v_dot = (k * u - self.drag_force(velocity)) / m

        return v_dot


if __name__ == "__main__":
    sys = UnderwaterVehicle()
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    print(f)
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))
