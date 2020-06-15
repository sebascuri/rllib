"""Vectorized Gym CartPole Environment."""

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.spaces.box import Box

from rllib.environment.vectorized.util import VectorizedEnv


class VectorizedCartPoleEnv(CartPoleEnv, VectorizedEnv):
    """Vectorized implementation of Cartpole with continuous actions."""

    def __init__(self, discrete=False):
        super().__init__()

        self.max_torque = 1.0
        self.discrete = discrete
        if not discrete:
            self.action_space = Box(
                low=-self.max_torque, high=self.max_torque, shape=(1,)
            )

    def step(self, action):
        """See `AcrobotEnv.step()'."""
        mass = self.total_mass
        pole_mass = self.masspole
        pole_length = self.polemass_length
        length = self.length
        grav = self.gravity

        bk = self.bk

        x, x_dot = self.state[..., 0], self.state[..., 1]
        theta, theta_dot = self.state[..., 2], self.state[..., 3]

        if self.discrete:
            force = self.force_mag * action.squeeze(-1)
        else:
            try:
                force = self.force_mag * action[..., 0]
            except IndexError:
                force = self.force_mag * action

        cos = bk.cos(theta)
        sin = bk.sin(theta)
        temp = (force + pole_length * theta_dot * theta_dot * sin) / mass

        thetaacc = (grav * sin - cos * temp) / (
            length * (4.0 / 3.0 - pole_mass * cos * cos / mass)
        )

        xacc = temp - pole_length * thetaacc * cos / mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = bk.stack((x, x_dot, theta, theta_dot), -1)
        done = (x < -self.x_threshold) + (x > self.x_threshold)
        done += (theta < -self.theta_threshold_radians) + (
            theta > self.theta_threshold_radians
        )

        reward = bk.ones(done.shape)
        return self.state, reward, done, {}


class DiscreteVectorizedCartPoleEnv(VectorizedCartPoleEnv):
    """Vectorized implementation of Cartpole with discrete actions."""

    def __init__(self):
        super().__init__(discrete=True)

    def step(self, action):
        """See `CartPoleEnv.step()'."""
        return super().step(2 * action - 1)
