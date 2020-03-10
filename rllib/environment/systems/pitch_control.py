"""Pitch Control Implementation."""

from rllib.environment.systems.ode_system import ODESystem
from rllib.environment.systems.linear_system import LinearSystem
from scipy import signal
import numpy as np


class PitchControl(ODESystem):
    """Pitch Control.

    For an explanation about the parameters look into the references.

    Parameters
    ----------
    omega: float, optional
    cld: float, optional
    cmld: float, optional
    cw: float, optional
    cm: float, optional.
    eta: float, optional
    step_size: float, optional

    References
    ----------
    Hafner, R., & Riedmiller, M. (2011).
    Reinforcement learning in feedback control. Machine learning.

    http://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=SystemModeling
    """

    def __init__(self, omega=56.7, cld=0.313, cmld=0.0139, cw=0.232, cm=0.426,
                 eta=.0875, step_size=0.01):
        self.omega = omega
        self.cld = cld
        self.cmld = cmld
        self.cw = cw
        self.cm = cm
        self.eta = eta
        super().__init__(
            func=self._ode,
            step_size=step_size,
            dim_action=1,
            dim_state=3)

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        ad : ndarray
            The discrete-time state matrix.
        bd : ndarray
            The discrete-time action matrix.

        """
        a = np.array([[-self.cld, self.omega, 0],
                      [-self.cmld, -self.cm, 0],
                      [0, self.omega, 0]]
                     )

        b = np.array([self.cw, self.eta * self.cw, 0]).reshape((-1, 2))

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
        alpha, q, theta = state
        u = action

        alpha_dot = -self.cld * alpha + self.omega * q + self.cw * u
        q_dot = -self.cmld * alpha - self.cm * q + self.eta * self.cw * u
        theta_dot = self.omega * q

        return np.array((alpha_dot, q_dot, theta_dot))
