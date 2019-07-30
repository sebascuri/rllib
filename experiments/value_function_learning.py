import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn.functional

import numpy as np
from rllib.environment import GymEnvironment
from rllib.agent import TDAgent
from rllib.value_function import NNValueFunction
from rllib.util import rollout_agent
from rllib.dataset import ExperienceReplay
import pickle

ENVIRONMENT = 'CartPole-v0'
AGENT = 'DDQN-Agent'
NUM_EPISODES = 10
LAYERS = [64, 64]
GAMMA = 0.99
LEARNING_RATE = 0.01
EPOCHS = 100
SEED = 0
BATCH_SIZE = 128
MAX_LEN = NUM_EPISODES * 200

torch.manual_seed(SEED)
np.random.seed(SEED)
environment = GymEnvironment(ENVIRONMENT, SEED)

with open('runs/{}_{}.pkl'.format(ENVIRONMENT, AGENT), 'rb') as file:
    training_agent = pickle.load(file)
policy = training_agent.policy
value_function = NNValueFunction(environment.dim_observation,
                                 num_states=environment.num_observation,
                                 layers=LAYERS)

criterion = torch.nn.functional.mse_loss
optimizer = torch.optim.Adam
memory = ExperienceReplay(max_len=MAX_LEN)
hyper_params = {
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'learning_rate': LEARNING_RATE,
    'epochs': EPOCHS
}
agent = TDAgent(policy, value_function, criterion, optimizer, memory, hyper_params)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

# plt.plot(training_agent.episodes_cumulative_rewards, label='Agent Learning')
# plt.plot(agent.episodes_cumulative_rewards, label='Executed Policy')
#
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Rewards')
# plt.legend(loc='best')
# plt.show()
states = torch.zeros(100, 4)
states[:, 0] = torch.linspace(-3, 3, 100)
plt.plot(states[:, 0].numpy(), value_function(states).detach().numpy())
plt.show()
