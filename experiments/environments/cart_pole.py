"""Python Script Template."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from rllib.agent import QLearningAgent
from rllib.algorithms import DDQN
from rllib.dataset import PrioritizedExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import EpsGreedy
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction

ENVIRONMENT = 'CartPole-v0'
NUM_EPISODES = 50
MILESTONES = [0, 20, NUM_EPISODES-1]
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
MOMENTUM = 0.1
WEIGHT_DECAY = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
policy = EpsGreedy(q_function, EPS_START, EPS_END, EPS_DECAY)


optimizer = torch.optim.SGD(q_function.parameters(), lr=LEARNING_RATE,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss
memory = PrioritizedExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)
# memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

agent = QLearningAgent(DDQN, q_function, policy, criterion, optimizer, memory,
                  target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)
rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES,
              milestones=MILESTONES)

with open('{}_{}.pkl'.format(environment.name, agent.name), 'wb') as file:
    pickle.dump(agent, file)

plt.plot(agent.logs['rewards'].episode_log)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)
