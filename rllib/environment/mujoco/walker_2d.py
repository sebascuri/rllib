"""Walker2d Environment with full observation."""
import gym.error

from .locomotion import LargeStateTermination, LocomotionEnv

try:
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    Walker2dEnv = object


class MBWalker2dEnv(LocomotionEnv, Walker2dEnv):
    """Walker2d Environment."""

    def __init__(self, ctrl_cost_weight=1e-3):
        self.base_mujoco_name = "Walker2d-v3"

        LocomotionEnv.__init__(
            self,
            dim_pos=1,
            dim_action=(6,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        Walker2dEnv.__init__(
            self,
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        self._termination_model = LargeStateTermination(
            z_dim=1,
            healthy_angle_range=self._healthy_angle_range,
            healthy_z_range=self._healthy_z_range,
        )
