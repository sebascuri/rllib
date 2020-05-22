"""Working example of DDPG."""
from itertools import chain

import numpy as np
import torch.nn.functional as func
import torch.optim

from rllib.util.training import train_agent, evaluate_agent
from rllib.agent import TD3Agent, DPGAgent
from rllib.dataset import ExperienceReplay, PrioritizedExperienceReplay
from rllib.environment import GymEnvironment
from rllib.exploration_strategies import GaussianNoise
from rllib.policy import FelixPolicy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

ENVIRONMENT = ['MountainCarContinuous-v0', 'Pendulum-v0'][0]
NUM_EPISODES = 25
MAX_STEPS = 2500
TARGET_UPDATE_FREQUENCY = 2
TARGET_UPDATE_TAU = 0.01
MEMORY_MAX_SIZE = 25000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e6
LAYERS = [64, 64]
SEED = 0
RENDER = True

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = FelixPolicy(environment.dim_state, environment.dim_action, deterministic=True,
                     tau=TARGET_UPDATE_TAU)
noise = GaussianNoise(ExponentialDecay(EPS_START, EPS_END, EPS_DECAY))
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
memory = PrioritizedExperienceReplay(max_len=MEMORY_MAX_SIZE)

optimizer = torch.optim.Adam(chain(policy.parameters(), q_function.parameters()),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
critic_optimizer = None
criterion = torch.nn.MSELoss

agent = DPGAgent(
    environment.name, q_function, policy, noise, criterion, optimizer=optimizer,
    memory=memory, batch_size=BATCH_SIZE,
    target_update_frequency=TARGET_UPDATE_FREQUENCY, exploration_episodes=1,
    gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, render=True)
evaluate_agent(agent, environment, 1, MAX_STEPS)
