"""Implementation of InvertedPendulum System."""


from rllib.environment.systems.ode_system import ODESystem
import numpy as np
from scipy import signal
from rllib.environment.systems.linear_system import LinearSystem


__all__ = ['InvertedPendulum']


class InvertedPendulum(ODESystem):
    """Inverted Pendulum system.

    Parameters
    ----------
    mass : float
    length : float
    friction : float
    gravity: float, optional
    step_size : float, optional
        The duration of each time step.
    """

    def __init__(self, mass, length, friction, gravity=9.81, step_size=0.01):
        """Initialization; see `InvertedPendulum`."""
        self.mass = mass
        self.length = length
        self.friction = friction
        self.gravity = gravity

        super().__init__(
            ode=self._ode,
            step_size=step_size,
            dim_action=1,
            dim_state=2,
        )

    @property
    def inertia(self):
        """Return the inertia of the pendulum."""
        return self.mass * self.length ** 2

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        a = np.array([[0, 1],
                      [gravity / length, -friction / inertia]])

        b = np.array([[0],
                      [1 / inertia]])

        sys = signal.StateSpace(a, b, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.step_size)
        return LinearSystem(sysd.A, sysd.B)

    def _ode(self, _, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state : ndarray
        action : ndarray

        Returns
        -------
        x_dot : Tensor
            The derivative of the state.

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        angle, angular_velocity = state

        x_ddot = (gravity / length * np.sin(angle)
                  + action / inertia
                  - friction / inertia * angular_velocity)

        return np.array((angular_velocity, x_ddot))
