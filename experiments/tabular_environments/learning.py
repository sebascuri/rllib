from rllib.environment import EasyGridWorld
from rllib.policy import RandomPolicy
from rllib.value_function import TabularValueFunction
from rllib.agent import TDAgent, MCAgent, OnLineTDLearning
from rllib.algorithms import policy_evaluation
from rllib.util import rollout_agent

import torch
import torch.nn.functional
import torch.optim
import matplotlib.pyplot as plt


def evaluate_episode(values, logs, true_values):
    errors = []
    for parameters in logs:
        values.parameters = parameters
        errors.append((values.table - true_values.table).pow(2).mean().item())

    return errors


environment = EasyGridWorld()
GAMMA = 0.9
EPS = 1e-6
LEARNING_RATE = 0.1
EPISODE_LENGTH = 100
NUM_EPISODES = 50
LAMBDA = 0.7

criterion = torch.nn.functional.mse_loss
optimizer = torch.optim.Adam
hyper_params = {'learning_rate': LEARNING_RATE}

policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                      num_actions=environment.num_actions)
true_value_function = policy_evaluation(policy, environment, GAMMA)
print(true_value_function.table)


fig, (ax1, ax2) = plt.subplots(2, 1)
for name, Agent in {'td0': TDAgent, 'mc': MCAgent, 'td.7': OnLineTDLearning}.items():
    value_function = TabularValueFunction(num_states=environment.num_states)
    if Agent == OnLineTDLearning:
        agent = Agent(policy, value_function, criterion, optimizer, hyper_params,
                      LAMBDA, GAMMA, EPISODE_LENGTH)
    else:
        agent = Agent(policy, value_function, criterion, optimizer, hyper_params,
                      GAMMA, EPISODE_LENGTH)

    rollout_agent(environment, agent, num_episodes=NUM_EPISODES)
    print(value_function.table)
    errors = (evaluate_episode(value_function, agent.logs['value_function'],
                               true_value_function))
    ax1.plot(errors, label=name)
    ax2.plot(agent.logs['td_error'], label=name)
#
#
ax1.set_ylabel('Value Function Error')
ax1.legend(loc='best')

ax2.set_ylabel('TD-Error')
ax2.set_xlabel('Episodes')

plt.show()
