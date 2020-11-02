"""Working example of DDPG."""
import numpy as np
import torch.optim

from rllib.agent import DPGAgent, TD3Agent  # noqa: F401
from rllib.dataset import ExperienceReplay, PrioritizedExperienceReplay  # noqa: F401
from rllib.environment import GymEnvironment
from rllib.util.parameter_decay import ExponentialDecay
from rllib.util.training.agent_training import evaluate_agent, train_agent

ENVIRONMENT = ["MountainCarContinuous-v0", "Pendulum-v0"][0]
NUM_EPISODES = 25
MAX_STEPS = 2500
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e6
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
noise = ExponentialDecay(EPS_START, EPS_END, EPS_DECAY)
agent = DPGAgent.default(environment, exploration_noise=noise, gamma=GAMMA)

train_agent(
    agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, render=True
)
evaluate_agent(agent, environment, 1, MAX_STEPS)
