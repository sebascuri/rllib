"""Mujoco Reacher environment from https://github.com/kchua/handful-of-trials."""

import os

import numpy as np
import torch
from gym import utils
from torch import cos, sin

from rllib.reward.state_action_reward import StateActionReward


class ReacherReward(StateActionReward):
    """Reward of Reacher Environment."""

    dim_action = (7,)
    length_scale = 0.45

    def __init__(self, goal, ctrl_cost_weight=0.001, sparse=False):
        super().__init__(ctrl_cost_weight, sparse, goal=goal)

    def copy(self):
        """Get copy of Reacher Reward."""
        return ReacherReward(
            ctrl_cost_weight=self.ctrl_cost_weight, sparse=self.sparse, goal=self.goal
        )

    def state_sparse_reward(self, dist_to_target):
        """Sparse reacher reward model."""
        scale = self.length_scale
        return torch.exp(-torch.square(dist_to_target).sum(-1) / (scale ** 2))

    def state_non_sparse_reward(self, dist_to_target):
        """Non-sparse reacher reward model."""
        return -torch.square(dist_to_target).sum(-1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        goal = state[..., -3:]
        dist_to_target = self.get_ee_position(state) - goal
        if self.sparse:
            return self.state_sparse_reward(dist_to_target)
        else:
            return self.state_non_sparse_reward(dist_to_target)

    @staticmethod
    def get_ee_position(state):
        """Get the end effector position."""
        theta1, theta2 = state[..., 0], state[..., 1]
        theta3, theta4 = state[..., 2:3], state[..., 3:4]
        theta5, theta6 = state[..., 4:5], state[..., 5:6]

        rot_axis = torch.stack(
            [cos(theta2) * cos(theta1), cos(theta2) * sin(theta1), -sin(theta2)], -1
        )
        rot_perp_axis = torch.stack(
            [-sin(theta1), cos(theta1), torch.zeros_like(theta1)], -1
        )

        cur_end = torch.stack(
            [
                0.1 * cos(theta1) + 0.4 * cos(theta1) * cos(theta2),
                0.1 * sin(theta1) + 0.4 * sin(theta1) * cos(theta2) - 0.188,
                -0.4 * sin(theta2),
            ],
            -1,
        )
        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
            x = rot_axis * cos(hinge)
            y = sin(hinge) * sin(roll) * rot_perp_axis
            z = -sin(hinge) * cos(roll) * perp_all_axis

            new_rot_axis = x + y + z
            new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)

            norm = torch.sqrt(torch.square(new_rot_perp_axis).sum(-1))
            new_rot_perp_axis[norm < 1e-30] = rot_perp_axis[norm < 1e-30]

            new_rot_perp_axis /= torch.sqrt(torch.square(new_rot_perp_axis).sum(-1))[
                ..., None
            ]

            rot_axis, rot_perp_axis = new_rot_axis, new_rot_perp_axis
            cur_end = cur_end + length * new_rot_axis

        return cur_end


try:
    from gym.envs.mujoco import mujoco_env

    class MBReacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """Reacher environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, goal_at_obs=True, ctrl_cost_weight=0.001, sparse=False):
            self._reward_model = ReacherReward(
                ctrl_cost_weight=ctrl_cost_weight, sparse=sparse, goal=None
            )
            self.action_magnitude = 20
            self.goal = None
            self.goal_at_obs = goal_at_obs
            utils.EzPickle.__init__(self)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(self, "%s/assets/reacher3d.xml" % dir_path, 2)
            self.action_space.high = np.ones_like(self.action_space.high)
            self.action_space.low = -np.ones_like(self.action_space.low)

        def reward_model(self):
            """Get reward model."""
            return self._reward_model.copy()

        def step(self, action):
            """See `AbstractEnvironment.step()'."""
            action = action * self.action_magnitude
            obs = self._get_obs()
            reward = self._reward_model(obs, action)[0].item()
            self.do_simulation(action, self.frame_skip)
            next_obs = self._get_obs()
            done = False
            return next_obs, reward, done, self._reward_model.info

        def viewer_setup(self):
            """Set-up the viewer."""
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 2.5
            self.viewer.cam.elevation = -30
            self.viewer.cam.azimuth = 270

        def reset_model(self):
            """Reset the model."""
            qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
            qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
            qvel[-3:] = 0
            self.set_state(qpos, qvel)
            if self.goal_at_obs:
                self.goal = None
            else:
                self.goal = qpos[-3:]
            return self._get_obs()

        def _get_obs(self):
            if self.goal_at_obs:
                return np.concatenate(
                    [
                        self.sim.data.qpos.flat[:-3],
                        self.sim.data.qvel.flat[:-3],
                        self.sim.data.qpos.flat[-3:],
                    ]
                )
            else:
                return np.concatenate(
                    [self.sim.data.qpos.flat[:-3], self.sim.data.qvel.flat[:-3]]
                )


except Exception:  # Mujoco not installed.
    pass
