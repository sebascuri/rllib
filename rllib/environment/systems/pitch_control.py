"""Pitch Control Implementation."""

import numpy as np

from rllib.environment.systems.ode_system import ODESystem


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

    def __init__(
        self,
        omega=56.7,
        cld=0.313,
        cmld=0.0139,
        cw=0.232,
        cm=0.426,
        eta=0.0875,
        step_size=0.01,
    ):
        self.omega = omega
        self.cld = cld
        self.cmld = cmld
        self.cw = cw
        self.cm = cm
        self.eta = eta
        super().__init__(
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(3,)
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
        # Physical dynamics
        alpha, q, theta = state
        (u,) = action

        alpha_dot = -self.cld * alpha + self.omega * q + self.cw * u
        q_dot = -self.cmld * alpha - self.cm * q + self.eta * self.cw * u
        theta_dot = self.omega * q

        return np.array([alpha_dot, q_dot, theta_dot])


if __name__ == "__main__":
    sys = PitchControl()
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    print(f)
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))
