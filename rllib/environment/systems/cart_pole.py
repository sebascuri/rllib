"""Implementation of CarPole System."""

import numpy as np

from rllib.environment import SystemEnvironment
from rllib.environment.systems.ode_system import ODESystem
from rllib.util.utilities import get_backend


class CartPole(ODESystem):
    """Cart with mounted inverted pendulum.

    Parameters
    ----------
    pendulum_mass : float
    cart_mass : float
    length : float
    rot_friction : float, optional
    gravity: float, optional
    step_size : float, optional
    """

    def __init__(
        self,
        pendulum_mass,
        cart_mass,
        length,
        rot_friction=0.0,
        gravity=9.81,
        step_size=0.01,
    ):
        """Initialization; see `CartPole`."""
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.rot_friction = rot_friction
        self.gravity = gravity

        super().__init__(
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(4,)
        )

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
        bk = get_backend(state)
        # Physical dynamics
        pendulum_mass = self.pendulum_mass
        cart_mass = self.cart_mass
        length = self.length
        b = self.rot_friction
        g = self.gravity

        total_mass = pendulum_mass + cart_mass

        x, theta, v, omega = state

        x_dot = v
        theta_dot = omega

        det = length * (cart_mass + pendulum_mass * bk.sin(theta) ** 2)
        v_dot = (
            (
                action
                - pendulum_mass * length * (omega ** 2) * bk.sin(theta)
                - b * omega * bk.cos(theta)
                + 0.5 * pendulum_mass * g * length * bk.sin(2 * theta)
            )
            * length
            / det
        )
        omega_dot = (
            action * bk.cos(theta)
            - 0.5 * pendulum_mass * length * (omega ** 2) * bk.sin(2 * theta)
            - b * total_mass * omega / (pendulum_mass * length)
            + total_mass * g * bk.sin(theta)
        ) / det

        return np.array((x_dot, theta_dot, v_dot, omega_dot))


class CartPoleEnv(SystemEnvironment):
    """CartPole Environment."""

    def __init__(self, pendulum_mass=1, cart_mass=1, length=1):
        super().__init__(CartPole(pendulum_mass, cart_mass, length))


if __name__ == "__main__":
    sys = CartPole(1, 1, 0.1)
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))
