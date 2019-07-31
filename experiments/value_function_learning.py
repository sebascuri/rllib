import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn.functional

import numpy as np
from rllib.environment import GymEnvironment
from rllib.agent import TDAgent, MCAgent
from rllib.value_function import NNValueFunction
from rllib.util import rollout_agent
from rllib.dataset import ExperienceReplay
import pickle

ENVIRONMENT = 'CartPole-v0'
AGENT = 'DDQN-Agent'
NUM_EPISODES = 20
LAYERS = []
GAMMA = 0.99
LEARNING_RATE = 0.0001
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
environment = GymEnvironment(ENVIRONMENT, SEED)

with open('runs/{}_{}.pkl'.format(ENVIRONMENT, AGENT), 'rb') as file:
    training_agent = pickle.load(file)
policy = training_agent.policy
value_function = NNValueFunction(environment.dim_observation,
                                 num_states=environment.num_observations,
                                 layers=LAYERS)

criterion = torch.nn.functional.mse_loss
optimizer = torch.optim.SGD
hyper_params = {
    'gamma': GAMMA,
    'learning_rate': LEARNING_RATE,
}
for Agent in [TDAgent, MCAgent]:
    agent = Agent(policy, value_function, criterion, optimizer, hyper_params)

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
    plt.plot(states[:, 0].numpy(), value_function(states).detach().numpy(),
             label=str(agent))

plt.legend(loc='best')
plt.show()
