"""Hopper Environment with full observation."""
import gym.error

from .locomotion import LocomotionEnv

try:
    from gym.envs.mujoco.hopper_v3 import HopperEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HopperEnv = object


class MBHopperEnv(LocomotionEnv, HopperEnv):
    """Hopper Environment."""

    def __init__(self, action_cost=1e-3):
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        HopperEnv.__init__(
            self,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
