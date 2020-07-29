"""Implementation of CarPole System."""

import numpy as np

from rllib.environment.systems.ode_system import ODESystem
from rllib.util.utilities import get_backend


class Cart1d(ODESystem):
    r"""Cart 1-D.

    A cart dynamics is described with a position and velocity x = [p, v]
    The action is an acceleration. The state dynamics is given by:

    ..math :: dx/dt = [v, a]
    """

    def __init__(self, step_size=0.01, max_action=1.0):
        """Initialization; see `CartPole`."""
        self.max_action = max_action
        super().__init__(
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(2,)
        )

    def _ode(self, t, state, action):
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
        velocity = state[..., 1]
        x_dot = velocity
        if bk == np:
            acceleration = bk.clip(action[..., 0], -self.max_action, self.max_action)
        else:
            acceleration = bk.clamp(action[..., 0], -self.max_action, self.max_action)
        v_dot = acceleration

        return bk.stack((x_dot, v_dot), -1)


if __name__ == "__main__":
    sys = Cart1d(step_size=0.1)
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))

    sys.state = np.zeros(sys.dim_state)
    print(sys.step(np.ones(sys.dim_action)))
    print(sys.step(np.ones(sys.dim_action)))
