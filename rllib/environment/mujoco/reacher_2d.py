"""Reacher 2d Environment with full observation."""
import gym.error
import numpy as np
import torch

from rllib.reward.state_action_reward import StateActionReward

try:
    from gym.envs.mujoco.reacher import ReacherEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    ReacherEnv = object


class ReacherReward(StateActionReward):
    """Reacher Reward model."""

    dim_action = (2,)

    def __init__(self, goal, ctrl_cost_weight=1.0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, goal=goal)

    def copy(self):
        """Copy reward model."""
        return ReacherReward(ctrl_cost_weight=self.ctrl_cost_weight, goal=self.goal)

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        dist_to_target = state[..., -3:]
        return -torch.sqrt((dist_to_target ** 2).sum(-1))


class MBReacherEnv(ReacherEnv):
    """Reacher Environment."""

    def __init__(self, goal_at_obs=True, ctrl_cost_weight=1.0):
        self.base_mujoco_name = "Reacher-v2"
        self._reward_model = ReacherReward(ctrl_cost_weight=ctrl_cost_weight, goal=None)
        self.goal_at_obs = goal_at_obs
        super().__init__()
        if self.goal_at_obs:
            self.goal = None
        else:
            self._reward_model.set_goal(self.goal)

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def reset_model(self):
        """Reset model."""
        obs = super().reset_model()
        self.goal = None
        return obs

    def step(self, a):
        """See `AbstractEnvironment.step()'."""
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()
        done = False
        return next_obs, reward, done, self._reward_model.info

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        if self.goal_at_obs:
            return np.concatenate(
                [
                    np.cos(theta),
                    np.sin(theta),
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:2],
                    self.get_body_com("fingertip") - self.get_body_com("target"),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.cos(theta),
                    np.sin(theta),
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:2],
                ]
            )
