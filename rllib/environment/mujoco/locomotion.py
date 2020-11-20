"""Python Script Template."""
import gym
import numpy as np
import torch

from rllib.model import AbstractModel
from rllib.reward.locomotion_reward import LocomotionReward

try:
    from gym.envs.mujoco.humanoid_v3 import HumanoidEnv, mass_center
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    SwimmerEnv, mass_center = None, None


class LargeStateTermination(AbstractModel):
    """Hopper Termination Function."""

    def __init__(
        self,
        z_dim=None,
        healthy_state_range=(-100, 100),
        healthy_z_range=(-np.inf, np.inf),
        healthy_angle_range=(-np.inf, np.inf),
    ):
        super().__init__(dim_state=(), dim_action=(), model_kind="termination")
        self.z_dim = z_dim
        self.healthy_state_range = healthy_state_range
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range

    def copy(self):
        """Get copy of termination model."""
        return LargeStateTermination(
            z_dim=self.z_dim,
            healthy_state_range=self.healthy_state_range,
            healthy_z_range=self.healthy_state_range,
            healthy_angle_range=self.healthy_angle_range,
        )

    @staticmethod
    def in_range(state, min_max_range):
        """Check if state is in healthy range."""
        min_state, max_state = min_max_range
        return (min_state < state) * (state < max_state)

    def is_healthy(self, state):
        """Check if state is healthy."""
        if self.z_dim is None:
            return self.in_range(state, min_max_range=self.healthy_state_range).all(-1)
        z = state[..., self.z_dim]
        angle = state[..., self.z_dim + 1]
        other = state[..., self.z_dim + 1 :]

        return (
            self.in_range(z, min_max_range=self.healthy_z_range)
            * self.in_range(angle, min_max_range=self.healthy_angle_range)
            * self.in_range(other, min_max_range=self.healthy_state_range).all(-1)
        )

    def forward(self, state, action, next_state=None):
        """Return termination model logits."""
        if not isinstance(state, torch.Tensor):
            return ~self.is_healthy(state)
        done = ~self.is_healthy(state)
        return (
            torch.zeros(*done.shape, 2)
            .scatter_(dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf"))
            .squeeze(-1)
        )


class LocomotionEnv(object):
    """Base Locomotion environment. Is a hack to avoid repeated code."""

    def __init__(
        self,
        dim_pos,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
    ):
        self.dim_pos = dim_pos
        self.prev_pos = np.zeros(dim_pos)
        self._reward_model = LocomotionReward(
            dim_action=dim_action,
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=forward_reward_weight,
            healthy_reward=healthy_reward,
        )
        self.reward_range = ()
        self._termination_model = LargeStateTermination()

    def step(self, action):
        """See gym.Env.step()."""
        obs = self._get_obs()
        reward = self._reward_model(obs, action)[0].item()
        done = self._termination_model(obs, action)

        if isinstance(self, HumanoidEnv):
            self.prev_pos = mass_center(self.model, self.sim)
        else:
            self.prev_pos = self.sim.data.qpos[: self.dim_pos].copy()
        self.do_simulation(action, self.frame_skip)

        next_obs = self._get_obs()
        info = self._reward_model.info
        info.update(x_position=self.prev_pos[0], x_velocity=obs[0])
        if self.dim_pos == 2:
            info.update(
                y_poisition=self.prev_pos[1],
                y_velocity=obs[1],
                distance_from_origin=np.linalg.norm(self.prev_pos, ord=2),
            )

        return next_obs, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        if isinstance(self, HumanoidEnv):
            x_position = mass_center(self.model, self.sim)
        else:
            x_position = position[: self.dim_pos]
        forward_vel = (x_position - self.prev_pos) / self.dt
        return np.concatenate((forward_vel, position[self.dim_pos :], velocity)).ravel()

    def reset_model(self):
        """Reset model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        qpos[: self.dim_pos] = np.zeros(self.dim_pos).copy()
        self.prev_pos = -self.dt * qvel[: self.dim_pos].copy()
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def termination_model(self):
        """Get default termination model."""
        return LargeStateTermination()
