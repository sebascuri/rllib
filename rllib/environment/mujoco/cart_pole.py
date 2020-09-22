"""Mujoco CartPole environment from https://github.com/kchua/handful-of-trials."""
import os

import numpy as np
import torch
from gym import utils

from rllib.reward.state_action_reward import StateActionReward


class CartPoleReward(StateActionReward):
    r"""A cart-pole reward model implementation.

    The reward function is computed as:
    r(s, a) = e^(-(end-effector / length)^2) + action_reward.

    The action reward is computed from the state-action reward.
    """

    dim_action = (1,)

    def __init__(self, ctrl_cost_weight, pendulum_length):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.pendulum_length = pendulum_length

    def copy(self):
        """Get copy of reward model."""
        return CartPoleReward(
            ctrl_cost_weight=self.ctrl_cost_weight, pendulum_length=self.pendulum_length
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        end_effector = self._get_ee_pos(state[..., 0], state[..., 1])
        reward_state = torch.exp(
            -torch.square(end_effector).sum(-1) / (self.pendulum_length ** 2)
        )
        return reward_state

    def _get_ee_pos(self, x0, theta):
        sin, cos = torch.sin(theta), torch.cos(theta)
        return torch.stack(
            [x0 - self.pendulum_length * sin, -self.pendulum_length * (1 + cos)], -1
        )


try:
    from gym.envs.mujoco import mujoco_env

    class MBCartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """CartPole environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, ctrl_cost_weight=0.01, pendulum_length=0.6):
            self._reward_model = CartPoleReward(
                pendulum_length=pendulum_length, ctrl_cost_weight=ctrl_cost_weight
            )
            utils.EzPickle.__init__(self)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(self, f"{dir_path}/assets/cartpole.xml", 2)

        def step(self, action):
            """See `AbstractEnvironment.step()'."""
            ob = self._get_obs()
            reward = self._reward_model(ob, action)[0].item()
            self.do_simulation(action, self.frame_skip)
            next_obs = self._get_obs()
            done = False

            return next_obs, reward, done, self._reward_model.info

        def reward_model(self):
            """Get reward model."""
            return self._reward_model.copy()

        def reset_model(self):
            """Reset the model."""
            qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
            qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
            self.set_state(qpos, qvel)
            return self._get_obs()

        def _get_obs(self):
            return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

        def viewer_setup(self):
            """Set-up the viewer."""
            v = self.viewer
            v.cam.trackbodyid = 0
            v.cam.distance = self.model.stat.extent


except Exception:  # Mujoco not installed.
    pass
