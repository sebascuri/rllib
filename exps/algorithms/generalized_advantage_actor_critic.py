"""Working example of ACTOR-CRITIC."""

import numpy as np
import torch

from rllib.agent import GAACAgent
from rllib.environment import GymEnvironment
from rllib.util.training import evaluate_agent, train_agent

ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200
NUM_ROLLOUTS = 4
NUM_EPISODES = 1000
TARGET_UPDATE_FREQUENCY = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)


agent = GAACAgent.default(environment)
train_agent(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, environment, 1, MAX_STEPS)
