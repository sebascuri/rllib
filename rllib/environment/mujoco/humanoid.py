"""Humanoid Environment with full observation."""
import gym.error

from .locomotion import LargeStateTermination, LocomotionEnv

try:
    from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HumanoidEnv = object


class MBHumanoidEnv(LocomotionEnv, HumanoidEnv):
    """Humanoid Environment."""

    def __init__(self, ctrl_cost_weight=0.1):
        self.base_mujoco_name = "Humanoid-v3"
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            dim_action=(17,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.25,
            healthy_reward=5.0,
        )
        HumanoidEnv.__init__(
            self,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=0.0,
            forward_reward_weight=1.25,
            healthy_reward=5.0,
        )

        self._termination_model = LargeStateTermination(
            z_dim=2,
            healthy_z_range=self._healthy_z_range,
            healthy_state_range=(-1000, 1000),
        )
