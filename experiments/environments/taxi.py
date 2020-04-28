"""Taxi solution."""
import pickle
import torch

from rllib.agent import QLearningAgent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import EpsGreedy
from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNQFunction

ENVIRONMENT = 'Taxi-v3'
NUM_EPISODES = 200
MILESTONES = [0, 50, NUM_EPISODES - 1]
MAX_STEPS = 2000
TARGET_UPDATE_FREQUENCY = 1
TARGET_UPDATE_TAU = 1
MEMORY_MAX_SIZE = 2000
BATCH_SIZE = 16
LEARNING_RATE = 0.5
MOMENTUM = 0
WEIGHT_DECAY = 0
GAMMA = 0.99
EPSILON = 0.1
LAYERS = []
SEED = 0
RENDER = True

environment = GymEnvironment(ENVIRONMENT, SEED)
q_function = NNQFunction(
    dim_state=environment.dim_state, dim_action=environment.dim_action,
    num_states=environment.num_states, num_actions=environment.num_actions,
    layers=LAYERS, biased_head=False, tau=1)
q_function.nn.head.weight.data = torch.ones_like(q_function.nn.head.weight)

policy = EpsGreedy(q_function, EPSILON)
optimizer = torch.optim.SGD(q_function.parameters(), lr=LEARNING_RATE,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

agent = QLearningAgent(
    environment.name, q_function, policy, criterion, optimizer,  memory,
    batch_size=BATCH_SIZE, target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=True)
train_agent(agent, environment, 1, MAX_STEPS, render=True, plot_flag=True)

# evaluate_agent(agent, environment, 1, MAX_STEPS)

# with open('{}_{}.pkl'.format(environment.name, agent.name), 'wb') as file:
#     pickle.dump(agent, file)
