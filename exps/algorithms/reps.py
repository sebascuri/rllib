"""Working example of REPS."""
import numpy as np
import torch

from rllib.agent import REPSAgent
from rllib.environment import GymEnvironment
from rllib.util.training import evaluate_agent, train_agent

ETA = 1.0
NUM_EPISODES = 100

GAMMA = 1
SEED = 0
ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)

agent = REPSAgent.default(environment, epsilon=ETA, regularization=True, gamma=GAMMA)
train_agent(agent, environment, NUM_EPISODES, MAX_STEPS + 1)
evaluate_agent(agent, environment, 1, MAX_STEPS + 1)
