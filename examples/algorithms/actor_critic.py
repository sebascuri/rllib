"""Working example of ACTOR-CRITIC."""

import numpy as np
import torch

from rllib.agent import A2CAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = "NChain-v0"
MAX_STEPS = 200
NUM_ROLLOUTS = 1
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
agent = A2CAgent.default(environment, gamma=GAMMA)

train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS)
