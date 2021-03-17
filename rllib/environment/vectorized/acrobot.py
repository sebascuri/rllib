"""Vectorized Gym Acrobot Environment."""
import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv
from gym.envs.classic_control.pendulum import angle_normalize
from gym.spaces.box import Box

from rllib.environment.vectorized.util import VectorizedEnv, rk4


class VectorizedAcrobotEnv(AcrobotEnv, VectorizedEnv):
    """Vectorized implementation of Acrobot with continuous actions."""

    def __init__(self, discrete=False):
        super().__init__()

        self.max_torque = 1.0
        if not discrete:
            self.action_space = Box(
                low=-self.max_torque, high=self.max_torque, shape=(1,)
            )

    def step(self, action):
        """See `AcrobotEnv.step()'."""
        s = self.state
        torque = self.clip(action, -self.max_torque, self.max_torque)

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.rand(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = self.cat((s, torque), -1)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[..., :4]  # omit action

        ns[..., 0] = angle_normalize(ns[..., 0])
        ns[..., 1] = angle_normalize(ns[..., 1])
        ns[..., 2] = self.clip(ns[..., 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[..., 3] = self.clip(ns[..., 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()

        reward = -self.bk.ones(terminal.shape) * (~terminal)
        return self._get_ob(), reward, terminal, {}

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        bk = self.bk
        g = 9.8

        s, a = s_augmented[..., :-1], s_augmented[..., -1]
        theta1, theta2, dtheta1, dtheta2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]

        d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * bk.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * bk.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * bk.cos(theta1 + theta2 - np.pi / 2.0)

        phi1 = phi2 - (
            m2 * l1 * lc2 * dtheta2 ** 2 * bk.sin(theta2)
            + 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * bk.sin(theta2)
            - (m1 * lc1 + m2 * l1) * g * bk.cos(theta1 - np.pi / 2)
        )

        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a
                + d2 / d1 * phi1
                - m2 * l1 * lc2 * dtheta1 ** 2 * bk.sin(theta2)
                - phi2
            ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return bk.stack(
            (dtheta1, dtheta2, ddtheta1, ddtheta2, bk.zeros_like(dtheta1)), -1
        )

    def set_state(self, observation):
        """Set state from a given observation."""
        self.state = self.bk.zeros_like(observation[..., :4])
        self.state[..., 0] = self.atan2(observation[..., 1], observation[..., 0])
        self.state[..., 1] = self.atan2(observation[..., 1], observation[..., 0])
        self.state[..., 2:] = observation[4:]

    def _get_ob(self):
        bk = self.bk

        theta1, theta2 = self.state[..., 0], self.state[..., 1]
        dtheta1, dtheta2 = self.state[..., 2], self.state[..., 3]
        return bk.stack(
            (
                bk.cos(theta1),
                bk.sin(theta1),
                bk.cos(theta2),
                bk.sin(theta2),
                dtheta1,
                dtheta2,
            ),
            -1,
        )

    def _terminal(self):
        bk = self.bk
        s = self.state
        return -bk.cos(s[..., 0]) - bk.cos(s[..., 1] + s[..., 0]) > 1.0


class DiscreteVectorizedAcrobotEnv(VectorizedAcrobotEnv):
    """Vectorized Implementation of Acrobot with discrete actions."""

    def __init__(self):
        super().__init__(discrete=True)

    def step(self, action):
        """See `AcrobotEnv.step()'."""
        return super().step(-1 + action)
