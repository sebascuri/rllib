"""Working example of Soft Q-Learning."""

import numpy as np
import torch.optim

from rllib.agent import SoftQLearningAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = ["NChain-v0", "CartPole-v0"][1]

NUM_EPISODES = 50
MAX_STEPS = 200
GAMMA = 0.99
TEMPERATURE = 0.2
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)

agent = SoftQLearningAgent.default(environment, gamma=GAMMA, temperature=TEMPERATURE)
train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS)
