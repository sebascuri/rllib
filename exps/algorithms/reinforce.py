"""Working example of REINFORCE."""

import numpy as np
import torch

from rllib.agent import REINFORCEAgent
from rllib.environment import GymEnvironment
from rllib.util.training import evaluate_agent, train_agent

ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200
NUM_EPISODES = 1000

GAMMA = 0.99
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
agent = REINFORCEAgent.default(environment, gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, environment, 1, MAX_STEPS)
