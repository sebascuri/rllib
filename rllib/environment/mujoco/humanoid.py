"""Humanoid Environment with full observation."""
import gym.error
import numpy as np

try:
    from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HumanoidEnv = object


class MBHumanoidEnv(HumanoidEnv):
    """Humanoid Environment."""

    def __init__(self, action_cost=0.1):
        self.prev_pos = np.zeros(2)
        super().__init__(ctrl_cost_weight=action_cost, contact_cost_weight=0.0)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        xy_velocity = (position[:2] - self.prev_pos) / self.dt
        self.prev_pos = position[:2]

        return np.concatenate((xy_velocity, position[2:], velocity)).ravel()
