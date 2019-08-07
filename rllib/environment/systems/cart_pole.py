"""Implementation of CarPole System."""

from rllib.environment.systems.ode_system import ODESystem
from rllib.environment.systems.linear_system import LinearSystem
from scipy import signal
import numpy as np


__all__ = ['CartPole']


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

    def __init__(self, pendulum_mass, cart_mass, length, rot_friction=0., gravity=9.81,
                 step_size=0.01):
        """Initialization; see `CartPole`."""
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.rot_friction = rot_friction
        self.gravity = gravity

        super().__init__(
            ode=self._ode,
            step_size=step_size,
            dim_action=1,
            dim_state=4)

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        ad : ndarray
            The discrete-time state matrix.
        bd : ndarray
            The discrete-time action matrix.

        """
        pendulum_mass = self.pendulum_mass
        cart_mass = self.cart_mass
        length = self.length
        b = self.rot_friction
        g = self.gravity

        inertia = pendulum_mass * cart_mass * length ** 2
        total_mass = pendulum_mass + cart_mass

        a = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, g * pendulum_mass / cart_mass, 0, -b / (cart_mass * length)],
                      [0, g * total_mass / (length * cart_mass),
                       0, -b * total_mass / inertia]])

        b = np.array([0, 0, 1 / cart_mass, 1 / (cart_mass * length)]).reshape((-1, 2))

        ad, bd, _, _, _ = signal.cont2discrete((a, b, 0, 0), self.step_size,
                                               method='zoh')
        return LinearSystem(ad, bd)

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
        pendulum_mass = self.pendulum_mass
        cart_mass = self.cart_mass
        length = self.length
        b = self.rot_friction
        g = self.gravity

        total_mass = pendulum_mass + cart_mass

        x, theta, v, omega = state

        x_dot = v
        theta_dot = omega

        det = length * (cart_mass + pendulum_mass * np.sin(theta) ** 2)
        v_dot = (action - pendulum_mass * length * (omega ** 2) * np.sin(theta)
                 - b * omega * np.cos(theta)
                 + 0.5 * pendulum_mass * g * length * np.sin(2 * theta)) * length / det
        omega_dot = (action * np.cos(theta)
                     - 0.5 * pendulum_mass * length * (omega ** 2) * np.sin(2 * theta)
                     - b * total_mass * omega / (pendulum_mass * length)
                     + total_mass * g * np.sin(theta)) / det

        return np.array((x_dot, theta_dot, v_dot, omega_dot))
