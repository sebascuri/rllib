"""Working example of MPO."""

import numpy as np
import torch

from rllib.agent import MPOAgent
from rllib.environment import GymEnvironment
from rllib.policy import FelixPolicy, NNPolicy  # noqa: F401
from rllib.util.neural_networks import zero_bias  # noqa: F401
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = ["Taxi-v3", "CartPole-v0", "Pendulum-v0"][2]

NUM_EPISODES = 500
MAX_STEPS = 1000

GAMMA = 0.99
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
environment = GymEnvironment(ENVIRONMENT, SEED)

agent = MPOAgent.default(environment, gamma=GAMMA)

train_agent(
    agent,
    environment,
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    print_frequency=1,
    render=True,
)
evaluate_agent(agent, environment, num_episodes=2, max_steps=MAX_STEPS)
