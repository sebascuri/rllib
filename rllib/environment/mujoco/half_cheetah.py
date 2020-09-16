"""Half-Cheetah Environment with full observation."""
import gym.error
import numpy as np

from rllib.reward.state_action_reward import StateActionReward

try:
    from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HalfCheetahEnv = object


class MBHalfCheetahEnv(HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, action_cost=0.1):
        self.prev_pos = np.zeros(1)
        super().__init__(ctrl_cost_weight=action_cost)

    def step(self, action):
        """See `HalfCheetahEnv.step()'."""
        this_obs = self._get_obs()
        self.prev_pos = self.sim.data.qpos[:1].copy()
        self.do_simulation(action, self.frame_skip)

        ctrl_cost = self.control_cost(action)
        x_velocity = this_obs[0]
        forward_reward = self._forward_reward_weight * x_velocity

        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": self.prev_pos[0],
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return self._get_obs(), reward, done, info

    def reset_model(self):
        """See `HalfCheetahEnv.reset_model()'."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        qpos[:1] = 0.0
        self.prev_pos = -self.dt * qvel[:1].copy()
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        forward_velocity = (position[:1] - self.prev_pos) / self.dt
        return np.concatenate((forward_velocity, position[1:], velocity)).ravel()

    @staticmethod
    def reward_model(action_cost_ratio):
        """Get reward model."""
        #

        class HalfCheetahReward(StateActionReward):
            """Half Cheetah Reward model."""

            dim_action = (6,)

            def __init__(self):
                super().__init__(action_cost_ratio=action_cost_ratio)

            def state_reward(self, state, next_state=None):
                """Return state reward."""
                return state[..., 0]

        return HalfCheetahReward()
