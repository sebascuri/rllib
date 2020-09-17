"""Humanoid Environment with full observation."""
import gym.error

from .locomotion import LargeStateTermination, LocomotionEnv

try:
    from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HumanoidEnv = object


class MBHumanoidEnv(LocomotionEnv, HumanoidEnv):
    """Humanoid Environment."""

    def __init__(self, action_cost=0.1):
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            ctrl_cost_weight=action_cost,
            forward_reward_weight=1.25,
            healthy_reward=5.0,
        )
        HumanoidEnv.__init__(
            self,
            ctrl_cost_weight=action_cost,
            contact_cost_weight=0.0,
            forward_reward_weight=1.25,
            healthy_reward=5.0,
        )

    def termination_model(self):
        """Get Termination Model."""
        return LargeStateTermination(
            z_dim=2,
            healthy_z_range=self._healthy_z_range,
            healthy_state_range=(-1000, 1000),
        )
