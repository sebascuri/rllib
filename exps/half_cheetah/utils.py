"""Python Script Template."""

from rllib.reward.state_action_reward import StateActionReward
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np


class HalfCheetahReward(StateActionReward):
    """Reward function of HalfCheetah Environment."""

    dim_action = (6,)

    def __init__(self, action_cost=0.1):
        super().__init__(action_cost)

    def state_reward(self, state, next_state=None):
        """See `AbstractReward.forward()'."""
        return state[..., 0]


class HalfCheetahEnvV2(HalfCheetahEnv):
    """Half Cheetah V2 environment for MBRL control.

    References
    ----------
    Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
    Deep reinforcement learning in a handful of trials using probabilistic dynamics
    models. NeuRIPS.

    https://github.com/kchua/handful-of-trials
    """

    def __init__(self, action_cost=0.1):
        self.prev_x_pos = 0.0
        super().__init__(ctrl_cost_weight=action_cost)

    def _get_obs(self):
        x_vel = (self.sim.data.qpos.flat[0] - self.prev_x_pos) / self.dt
        self.prev_x_pos = self.sim.data.qpos.flat[0]
        return np.concatenate(
            [np.array([x_vel]), self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat]
        )
