from rllib.environment import EasyGridWorld
from rllib.policy import RandomPolicy
from rllib.value_function import TabularValueFunction
from rllib.agent import TDAgent, MCAgent
from rllib.algorithms import policy_evaluation
from rllib.util import rollout_agent

import torch
import torch.nn.functional
import torch.optim


environment = EasyGridWorld()
GAMMA = 0.9
EPS = 1e-6
LEARNING_RATE = 0.1
EPISODE_LENGTH = 100
criterion = torch.nn.functional.mse_loss
optimizer = torch.optim.Adam
hyper_params = {'gamma': GAMMA, 'learning_rate': LEARNING_RATE,
                'episode_length': EPISODE_LENGTH}

policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                      num_actions=environment.num_actions)
print(policy_evaluation(policy, environment, GAMMA).table)

value_function = TabularValueFunction(num_states=environment.num_states)
agent = TDAgent(policy, value_function, criterion, optimizer, hyper_params)
rollout_agent(environment, agent, num_episodes=100)
print(agent._value_function.table)


agent = MCAgent(policy, value_function, criterion, optimizer, hyper_params)
rollout_agent(environment, agent, num_episodes=100)
print(agent._value_function.table)