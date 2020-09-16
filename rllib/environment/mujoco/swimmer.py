"""Swimmer Environment with full observation."""
import gym.error

from .locomotion import LocomotionEnv

try:
    from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    SwimmerEnv = object


class MBSwimmerEnv(LocomotionEnv, SwimmerEnv):
    """Swimmer Environment."""

    def __init__(self, action_cost=1e-4):
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=0.0,
        )
        SwimmerEnv.__init__(
            self, ctrl_cost_weight=action_cost, forward_reward_weight=1.0
        )
