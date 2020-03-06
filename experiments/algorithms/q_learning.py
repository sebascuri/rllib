import matplotlib.pyplot as plt
from rllib.agent import QLearningAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction, TabularQFunction
from rllib.dataset import ExperienceReplay
from rllib.policy import EpsGreedy, SoftMax, MellowMax
from rllib.environment import GymEnvironment
from rllib.algorithms.q_learning import QLearning, SemiGQLearning, DDQN, DQN
import numpy as np
import torch.nn.functional as func
import torch.optim
import pickle

# ENVIRONMENT = 'NChain-v0'
ENVIRONMENT = 'CartPole-v0'

NUM_EPISODES = 50
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
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

agent = QLearningAgent(DDQN, q_function, policy, criterion, optimizer, memory,
                       target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)
rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)

plt.plot(agent.logs['rewards'].episode_log)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)
