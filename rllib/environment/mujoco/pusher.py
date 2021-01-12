"""Pusher Environment with full observation."""
import gym.error
import numpy as np
import torch

from rllib.reward.state_action_reward import StateActionReward

try:
    from gym.envs.mujoco.pusher import PusherEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    PusherEnv = object


class PusherReward(StateActionReward):
    """Pusher Reward model."""

    dim_action = (7,)

    def __init__(self, goal, ctrl_cost_weight=0.1):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, goal=goal)

    def copy(self):
        """Copy reward model."""
        return PusherReward(ctrl_cost_weight=self.ctrl_cost_weight, goal=self.goal)

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        goal = state[..., -3:]
        end_effector = state[..., -6:-3]
        pluck = state[..., -9:-6]

        dist_to_ball = pluck - end_effector
        dist_to_goal = pluck - goal

        reward_near = -torch.sqrt((dist_to_ball ** 2).sum(-1))
        reward_dist = -torch.sqrt((dist_to_goal ** 2).sum(-1))
        return reward_dist + 0.5 * reward_near


class MBPusherEnv(PusherEnv):
    """Pusher Environment."""

    def __init__(self, goal_at_obs=True, ctrl_cost_weight=0.1):
        self.base_mujoco_name = "Pusher-v2"
        self._reward_model = PusherReward(ctrl_cost_weight=ctrl_cost_weight, goal=None)
        self.goal_at_obs = goal_at_obs
        super().__init__()
        if self.goal_at_obs:
            self.goal = None
        else:
            self._reward_model.set_goal(self.get_body_com("goal"))

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def reset_model(self):
        """Reset model."""
        out = super().reset_model()
        if self.goal_at_obs:
            self.goal = None
        else:
            self._reward_model.set_goal(self.get_body_com("goal"))
        return out

    def step(self, a):
        """See `AbstractEnvironment.step()'."""
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()
        done = False
        return next_obs, reward, done, self._reward_model.info

    def _get_obs(self):
        if self.goal_at_obs:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[:7],
                    self.sim.data.qvel.flat[:7],
                    self.get_body_com("object"),
                    self.get_body_com("tips_arm"),
                    self.get_body_com("goal"),
                ]
            )
        else:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[:7],
                    self.sim.data.qvel.flat[:7],
                    self.get_body_com("object"),
                ]
            )
