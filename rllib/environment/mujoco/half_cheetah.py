"""Half-Cheetah Environment with full observation."""
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class MBHalfCheetahEnv(HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, action_cost=0.1):
        self.prev_pos = np.zeros(1)
        super().__init__(ctrl_cost_weight=action_cost)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        forward_velocity = (position[:1] - self.prev_pos) / self.dt
        self.prev_pos = position[:1]

        return np.concatenate((forward_velocity, position[1:], velocity)).ravel()
