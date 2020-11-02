"""Working example of SARSA."""

import numpy as np
import torch.optim

from rllib.agent import ExpectedSARSAAgent, SARSAAgent  # noqa: F401
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = "NChain-v0"
ALGORITHM = "SARSA"

NUM_EPISODES = 500
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 1
TARGET_UPDATE_TAU = 0.01
BATCH_SIZE = 1  # Batch size doesn't bring much because the data are heavily correlated.
LEARNING_RATE = 1e-3

GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)

if ALGORITHM == "SARSA":
    agent = SARSAAgent.default(environment, gamma=GAMMA)
else:
    agent = ExpectedSARSAAgent.default(environment, gamma=GAMMA)  # type: ignore

train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS)
