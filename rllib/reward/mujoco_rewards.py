"""Rewards of Mujoco Environments."""

import torch
import numpy as np
from rllib.reward import AbstractReward
from rllib.util.utilities import get_backend
from abc import ABCMeta


class MujocoReward(AbstractReward, metaclass=ABCMeta):
    """Base class for mujoco rewards."""

    def __init__(self, action_cost=0.01):
        super().__init__()
        self.action_cost = action_cost

    def get_reward(self, reward_state, reward_control):
        """Get reward distribution from reward_state, reward_control tuple."""
        reward = reward_state + self.action_cost * reward_control
        return reward, torch.zeros(1)


class CartPoleReward(MujocoReward):
    """Reward of MBRL CartPole Environment."""

    def __init__(self, action_cost=0.01, pendulum_length=0.6):
        super().__init__(action_cost)
        self.length = pendulum_length

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(next_state)
        end_effector = self._get_ee_pos(next_state[..., 0], next_state[..., 1])

        reward_state = bk.exp(-(end_effector ** 2).sum(-1) / (self.length ** 2))
        reward_control = -(action ** 2).sum(-1)
        return self.get_reward(reward_state, reward_control)

    def _get_ee_pos(self, x0, theta):
        bk = get_backend(x0)
        sin, cos = bk.sin(theta), bk.cos(theta)
        return bk.stack([x0 - self.length * sin, -self.length * (1 + cos)], -1)


class HalfCheetahReward(MujocoReward):
    """Reward of MBRL HalfCheetah Environment."""

    def __init__(self, action_cost=0.1):
        super().__init__(action_cost)

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        reward_state = next_state[..., 0] - 0.0 * next_state[..., 2] ** 2
        reward_control = -(action ** 2).sum(-1)

        return self.get_reward(reward_state, reward_control)


class PusherReward(MujocoReward):
    """Reward of MBRL Pusher Environment."""

    def __init__(self, action_cost=0.1, goal=np.zeros(3)):
        super().__init__(action_cost)
        self.goal = goal

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(state)

        obj_pos = state[..., -3:]
        vec_1 = obj_pos - state[..., -6:-3]
        vec_2 = obj_pos - self.goal

        reward_near = - bk.abs(vec_1).sum(-1)
        reward_dist = - bk.abs(vec_2).sum(-1)

        reward_state = 1.25 * reward_dist + 0.5 * reward_near
        reward_control = -(action ** 2).sum(-1)

        return self.get_reward(reward_state, reward_control)


class ReacherReward(MujocoReward):
    """Reward of Reacher Environment."""

    def __init__(self, action_cost=0.1, goal=torch.zeros(3)):
        super().__init__(action_cost)
        self.goal = goal

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        dist_to_target = self.get_goal_distance(next_state)
        reward_state = -(dist_to_target ** 2).sum(-1)
        reward_control = -(action ** 2).sum(-1)
        return self.get_reward(reward_state, reward_control)

    def get_goal_distance(self, state):
        """Get the distance between the end effector and the goal."""
        bk = get_backend(state)
        theta1, theta2 = state[..., 0], state[..., 1]
        theta3, theta4 = state[..., 2:3], state[..., 3:4]
        theta5, theta6 = state[..., 4:5], state[..., 5:6]

        rot_axis = bk.stack([bk.cos(theta2) * bk.cos(theta1),
                             bk.cos(theta2) * bk.sin(theta1),
                             -bk.sin(theta2)], -1)
        rot_perp_axis = bk.stack([-bk.sin(theta1), bk.cos(theta1),
                                  bk.zeros_like(theta1)], -1)

        cur_end = bk.stack([
            0.1 * bk.cos(theta1) + 0.4 * bk.cos(theta1) * bk.cos(theta2),
            0.1 * bk.sin(theta1) + 0.4 * bk.sin(theta1) * bk.cos(theta2) - 0.188,
            -0.4 * bk.sin(theta2)
        ], -1)
        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            if bk is torch:
                perp_all_axis = torch.tensor(perp_all_axis)

            x = rot_axis * bk.cos(hinge)
            y = bk.sin(hinge) * bk.sin(roll) * rot_perp_axis
            z = -bk.sin(hinge) * bk.cos(roll) * perp_all_axis

            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            if bk is torch:
                new_rot_perp_axis = torch.tensor(new_rot_perp_axis)

            norm = bk.sqrt((new_rot_axis ** 2).sum(-1))
            new_rot_perp_axis[norm < 1e-30] = rot_perp_axis[norm < 1e-30]

            new_rot_perp_axis /= bk.sqrt((new_rot_perp_axis ** 2).sum(-1))[..., None]

            rot_axis, rot_perp_axis = new_rot_axis, new_rot_perp_axis
            cur_end = cur_end + length * new_rot_axis

        return cur_end - self.goal
