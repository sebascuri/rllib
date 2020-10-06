"""Inverted Pendulum Environment with full observation."""
import gym.error
import numpy as np
import torch

from rllib.reward.state_action_reward import StateActionReward

from .locomotion import LargeStateTermination

try:
    from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    InvertedDoublePendulumEnv = object


class InvertedDoublePendulumReward(StateActionReward):
    """Inverted Double-Pendulum model."""

    def __init__(self, ctrl_cost_weight=0.0):
        self.dim_action = (1,)
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def copy(self):
        """Copy reward model."""
        return InvertedDoublePendulumReward(ctrl_cost_weight=self.ctrl_cost_weight)

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        x, y = state[..., -2:]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2

        v1, v2 = state[..., -4:-2]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        return 10 * torch.ones_like(state[..., 0]) - dist_penalty - vel_penalty


class MBInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    """Inverted Double-Pendulum Environment."""

    def __init__(self, ctrl_cost_weight=0.0):
        self.base_mujoco_name = "InvertedDoublePendulum-v2"
        self._reward_model = InvertedDoublePendulumReward(
            ctrl_cost_weight=ctrl_cost_weight
        )
        self._termination_model = LargeStateTermination(
            z_dim=-1,
            healthy_state_range=(-np.inf, np.inf),
            healthy_z_range=(1.0, np.inf),
        )
        super().__init__()

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def termination_model(self):
        """Get reward model."""
        return self._termination_model.copy()

    def step(self, a):
        """See `AbstractEnvironment.step()'."""
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        if np.all(np.all(obs == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
            done = False
        else:
            done = self._termination_model(obs, a)
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()

        return next_obs, reward, done, self._reward_model.info

    def _get_obs(self):
        x, _, y = self.sim.data.site_xpos[0]

        return np.concatenate(
            [
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                np.clip(self.sim.data.qvel, -10, 10),
                np.array([x, y]),
            ]
        ).ravel()
