"""Half-Cheetah Environment with full observation."""
import gym.error

from .locomotion import LocomotionEnv

try:
    from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HalfCheetahEnv = object


class MBHalfCheetahEnv(LocomotionEnv, HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, ctrl_cost_weight=0.1):
        self.base_mujoco_name = "HalfCheetah-v3"
        LocomotionEnv.__init__(
            self,
            dim_pos=1,
            dim_action=(6,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=0.0,
        )
        HalfCheetahEnv.__init__(
            self, ctrl_cost_weight=ctrl_cost_weight, forward_reward_weight=1.0
        )
