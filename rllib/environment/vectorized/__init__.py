"""Vectorized Gym Environments."""
from gym.envs.registration import register

from .acrobot import DiscreteVectorizedAcrobotEnv, VectorizedAcrobotEnv
from .cartpole import DiscreteVectorizedCartPoleEnv, VectorizedCartPoleEnv
from .pendulum import VectorizedPendulumEnv

register(
    id="VContinuous-CartPole-v0",
    entry_point="rllib.environment.vectorized.cartpole:VectorizedCartPoleEnv",
)

register(
    id="VDiscrete-CartPole-v0",
    entry_point="rllib.environment.vectorized.cartpole:DiscreteVectorizedCartPoleEnv",
)

register(
    id="VContinuous-Acrobot-v0",
    entry_point="rllib.environment.vectorized.acrobot:VectorizedAcrobotEnv",
)

register(
    id="VDiscrete-Acrobot-v0",
    entry_point="rllib.environment.vectorized.acrobot:DiscreteVectorizedAcrobotEnv",
)

register(
    id="VPendulum-v0",
    entry_point="rllib.environment.vectorized.pendulum:VectorizedPendulumEnv",
)
