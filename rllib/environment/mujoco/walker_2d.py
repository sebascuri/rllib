"""Walker2d Environment with full observation."""
import gym.error
import numpy as np

try:
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    Walker2dEnv = object


class MBWalker2dEnv(Walker2dEnv):
    """Walker2d Environment."""

    def __init__(self, action_cost=1e-3):
        self.prev_pos = np.zeros(1)
        super().__init__(ctrl_cost_weight=action_cost)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        x_velocity = (position[:1] - self.prev_pos) / self.dt
        self.prev_pos = position[:1]

        return np.concatenate((x_velocity, position[2:], velocity)).ravel()
