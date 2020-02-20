import matplotlib.pyplot as plt
from rllib.agent.sarsa_agent import *
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction
from rllib.policy import EpsGreedy, SoftMax, MellowMax
from rllib.environment import GymEnvironment
import numpy as np
import torch.nn.functional as func
import torch.optim

ENVIRONMENT = 'CartPole-v0'

NUM_EPISODES = 500
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 1
TARGET_UPDATE_TAU = 0.99
BATCH_SIZE = 1  # Batch size doesn't bring much because the data is heavily correlated.
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
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
policy = EpsGreedy(q_function, EPS_START, EPS_END, EPS_DECAY)
optimizer = torch.optim.Adam(q_function.parameters, lr=LEARNING_RATE)
criterion = torch.nn.MSELoss

agent = DExpectedSARSAAgent(q_function, policy, criterion, optimizer,
                            target_update_frequency=TARGET_UPDATE_FREQUENCY,
                            gamma=GAMMA, batch_size=BATCH_SIZE)
rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)

plt.plot(agent.episodes_cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)
