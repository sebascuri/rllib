import matplotlib.pyplot as plt
from rllib.agent import REINFORCE
from rllib.environment import GymEnvironment
from rllib.util.rollout import rollout_agent
from rllib.policy import MLPPolicy
from rllib.value_function import NNValueFunction, NNQFunction
import torch
import numpy as np

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
policy = MLPPolicy(environment.dim_state, environment.dim_action,
                   num_states=environment.num_states,
                   num_actions=environment.num_actions,
                   layers=LAYERS)

value_function = NNValueFunction(environment.dim_state,
                                 num_states=environment.num_states, layers=LAYERS)

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)
value_optimizer = torch.optim.Adam(value_function.parameters(), lr=CRITIC_LEARNING_RATE)
criterion = torch.nn.MSELoss

agent = REINFORCE(policy=policy, policy_optimizer=policy_optimizer,
                  baseline=value_function, baseline_optimizer=value_optimizer,
                  criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)

plt.plot(agent.logs['rewards'].episode_log)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.ion()
plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)
