"""Reacher 2d Environment with full observation."""
import gym.error
import torch

from rllib.reward.state_action_reward import StateActionReward

try:
    from gym.envs.mujoco.reacher import ReacherEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    ReacherEnv = object


class ReacherReward(StateActionReward):
    """Reacher Reward model."""

    dim_action = (2,)

    def __init__(self, ctrl_cost_weight=1.0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def copy(self):
        """Copy reward model."""
        return ReacherReward(ctrl_cost_weight=self.ctrl_cost_weight)

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        dist_to_target = state[-3:]
        return torch.sqrt((dist_to_target ** 2).sum(-1))


class MBReacherEnv(ReacherEnv):
    """Reacher Environment."""

    def __init__(self, ctrl_cost_weight=1.0):
        self.base_mujoco_name = "Reacher-v2"
        self._reward_model = ReacherReward(ctrl_cost_weight=ctrl_cost_weight)
        super().__init__()

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def step(self, a):
        """See `AbstractEnvironment.step()'."""
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()
        done = False
        return next_obs, reward, done, self._reward_model.info
