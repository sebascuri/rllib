"""Ant Environment with full observation."""
import gym.error

from .locomotion import LargeStateTermination, LocomotionEnv

try:
    from gym.envs.mujoco.ant_v3 import AntEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    AntEnv = object


class MBAntEnv(LocomotionEnv, AntEnv):
    """Ant Environment."""

    def __init__(self, ctrl_cost_weight=0.1):
        self.base_mujoco_name = "Ant-v3"
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            dim_action=(8,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        AntEnv.__init__(
            self,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=0.0,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
        )
        self._termination_model = LargeStateTermination(
            z_dim=2, healthy_z_range=self._healthy_z_range
        )
