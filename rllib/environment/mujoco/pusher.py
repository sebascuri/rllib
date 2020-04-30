"""Mujoco Pusher environment from https://github.com/kchua/handful-of-trials."""

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Pusher environment for MBRL control.

    References
    ----------
    Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
    Deep reinforcement learning in a handful of trials using probabilistic dynamics
    models. NeuRIPS.

    https://github.com/kchua/handful-of-trials
    """

    def __init__(self, action_cost=0.1):
        self.action_cost = action_cost
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, f"{dir_path}/assets/pusher.xml", 4)
        utils.EzPickle.__init__(self)
        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])
        self.goal = self.get_body_com("goal")
        self.reset_model()

    def step(self, action: np.ndarray):
        """See `AbstractEnvironment.step()'."""
        obj_pos = self.get_body_com("object")  # type: np.ndarray
        vec_1 = obj_pos - self.get_body_com("tips_arm")  # type: np.ndarray
        vec_2 = obj_pos - self.get_body_com("goal")  # type: np.ndarray

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(action).sum()
        reward = 1.25 * reward_dist + self.action_cost * reward_ctrl + 0.5 * reward_near

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        """Set-up the viewer."""
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        """Reset the model."""
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.goal = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])
