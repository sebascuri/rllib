"""Working example of REINFORCE."""

import numpy as np
import torch

from rllib.util.training import train_agent, evaluate_agent
from rllib.agent import REINFORCEAgent
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

ENVIRONMENT = 'CartPole-v0'
MAX_STEPS = 200
NUM_EPISODES = 1000
TARGET_UPDATE_FREQUENCY = 1
NUM_ROLLOUTS = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-2

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(environment.dim_state, environment.dim_action,
                  num_states=environment.num_states,
                  num_actions=environment.num_actions,
                  layers=LAYERS)

value_function = NNValueFunction(environment.dim_state,
                                 num_states=environment.num_states, layers=LAYERS)

optimizer = torch.optim.Adam([
    {'params': policy.parameters(), 'lr': ACTOR_LEARNING_RATE},
    {'params': value_function.parameters(), 'lr': CRITIC_LEARNING_RATE}
])

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)
value_optimizer = torch.optim.Adam(value_function.parameters(), lr=CRITIC_LEARNING_RATE)
criterion = torch.nn.MSELoss

agent = REINFORCEAgent(environment.name, policy=policy, baseline=value_function,
                       optimizer=optimizer,
                       criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, environment, 1, MAX_STEPS)
