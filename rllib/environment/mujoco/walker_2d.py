"""Walker2d Environment with full observation."""
import gym.error

from .locomotion import LocomotionEnv

try:
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    Walker2dEnv = object


class MBWalker2dEnv(LocomotionEnv, Walker2dEnv):
    """Walker2d Environment."""

    def __init__(self, action_cost=1e-3):
        LocomotionEnv.__init__(
            self,
            dim_pos=1,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        Walker2dEnv.__init__(
            self,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
