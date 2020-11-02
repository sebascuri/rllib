"""Implementation of InvertedPendulum System."""

import os

import numpy as np
from gym.spaces import Box

from rllib.environment.systems.ode_system import ODESystem
from rllib.util.utilities import get_backend

try:
    from gym.envs.classic_control import rendering
except Exception:  # No display.
    pass


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
            func=self._ode, step_size=step_size, dim_action=(1,), dim_state=(2,)
        )
        self.viewer = None
        self.last_action = None

    @property
    def inertia(self):
        """Return the inertia of the pendulum."""
        return self.mass * self.length ** 2

    def render(self, mode="human"):
        """Render pendulum."""
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = os.path.dirname(rendering.__file__) + "/assets/clockwise.png"
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_action:
            self.imgtrans.scale = (-self.last_action / 2, np.abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

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
        bk = get_backend(state)
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        if bk == np:
            action = action.clip(-1.0, 1.0)
        else:
            action = action.clamp(-1.0, 1.0)

        self.last_action = action[0]

        angle, angular_velocity = state

        x_ddot = (
            gravity / length * bk.sin(angle)
            + action[..., 0] / inertia
            - friction / inertia * angular_velocity
        )

        return np.array((angular_velocity, x_ddot))

    @property
    def action_space(self):
        """Return action space."""
        return Box(np.array([-1.0]), np.array([1.0]))

    @property
    def observation_space(self):
        """Return observation space."""
        return Box(np.array([-np.pi, -0.05]), np.array([np.pi, 0.05]))


if __name__ == "__main__":
    sys = InvertedPendulum(1, 1, 0.1)
    f = sys.func(None, np.ones(sys.dim_state), np.ones(sys.dim_action))
    print(f)
    sys.linearize()
    sys.linearize(np.ones(sys.dim_state), np.ones(sys.dim_action))
