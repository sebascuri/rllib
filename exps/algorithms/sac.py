"""Working example of SAC."""

import numpy as np
import torch.optim

from rllib.agent import SACAgent
from rllib.dataset import ExperienceReplay, PrioritizedExperienceReplay  # noqa: F401
from rllib.environment import GymEnvironment
from rllib.util.training import evaluate_agent, train_agent

ENVIRONMENT = ["MountainCarContinuous-v0", "Pendulum-v0"][1]
NUM_EPISODES = 40
MAX_STEPS = 1000
GAMMA = 0.99
SEED = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)

agent = SACAgent.default(environment, eta=1.0, regularization=True, gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, print_frequency=1, render=True)
evaluate_agent(agent, environment, 1, MAX_STEPS)
