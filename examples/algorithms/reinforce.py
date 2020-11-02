"""Working example of REINFORCE."""

import numpy as np
import torch

from rllib.agent import REINFORCEAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200
NUM_EPISODES = 1000

GAMMA = 0.99
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
agent = REINFORCEAgent.default(environment, gamma=GAMMA)

train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS)
