"""Magnetic Levitation Implementation."""

import numpy as np

from rllib.environment import SystemEnvironment
from rllib.environment.systems.ode_system import ODESystem


class MagneticLevitation(ODESystem):
    """Magnetic Levitation Control.

    Parameters
    ----------
    mass: float, optional
        Mass of Ball.
    resistance: float, optional
        Resistance of circuit.
    x_inf: float, optional
        Impedance of circuit.
    l_inf: float, optional
        Inductance of circuit.
    ksi: float, optional
        Inductance permeability.
    max_action: float, optional
        Maximum action.
    gravity: float, optional
        Gravity acceleration.
    step_size: float, optional
        Integration step-time.

    References
    ----------
    Hafner, R., & Riedmiller, M. (2011).
     Reinforcement learning in feedback control. Machine learning.
    """

    def __init__(
        self,
        mass=0.8,
        resistance=11.68,
        x_inf=0.007,
        l_inf=0.80502,
        ksi=0.001599,
        max_action=60,
        gravity=9.81,
        step_size=0.01,
    ):
        self.mass = mass
        self.resistance = resistance
        self.x_inf = x_inf
        self.l_inf = l_inf
        self.ksi = ksi
        self.max_action = max_action
        self.gravity = gravity

        super().__init__(
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(3,)
        )

    def alpha(self, x1, x2, x3):
        """Compute alpha coefficient."""
        x = self.x_inf + x1
        return self.gravity - self.ksi * x3 ** 2 / (2 * self.mass * x ** 2)

    def beta(self, x1, x2, x3):
        """Compute beta coefficient."""
        x = self.x_inf + x1
        return (
            x3
            * (self.ksi * x2 - self.resistance * x ** 2)
            / (self.ksi * x + self.l_inf * x ** 2)
        )

    def gamma(self, x1, x2, x3):
        """Compute gamma coefficient."""
        x = self.x_inf + x1
        return x / (self.ksi + self.l_inf * x)

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
        x1, x2, x3 = state

        alpha = self.alpha(x1, x2, x3)
        beta = self.beta(x1, x2, x3)
        gamma = self.gamma(x1, x2, x3)

        x1_dot = x2
        x2_dot = alpha
        x3_dot = beta + gamma * action

        return np.array([x1_dot, x2_dot, x3_dot])


class MagneticLevitationEnv(SystemEnvironment):
    """Magnetic Levitation Environment."""

    def __init__(self):
        super().__init__(MagneticLevitation())  # TODO: Add reward function.


if __name__ == "__main__":
    sys = MagneticLevitation()
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    print(f)
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))
