"""Ant Environment with full observation."""
import gym.error

from .locomotion import LocomotionEnv

try:
    from gym.envs.mujoco.ant_v3 import AntEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    AntEnv = object


class MBAntEnv(LocomotionEnv, AntEnv):
    """Ant Environment."""

    def __init__(self, action_cost=0.1):
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        AntEnv.__init__(
            self,
            ctrl_cost_weight=action_cost,
            contact_cost_weight=0.0,
            healthy_reward=1.0,
        )
