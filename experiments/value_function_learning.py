import matplotlib.pyplot as plt
import torch
import numpy as np
from rllib.environment import GymEnvironment
from rllib.util import rollout_policy
import pickle

ENVIRONMENT = 'CartPole-v0'
AGENT = 'DDQN-Agent'
NUM_EPISODES = 200
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
environment = GymEnvironment(ENVIRONMENT, SEED)

with open('runs/{}_{}.pkl'.format(ENVIRONMENT, AGENT), 'rb') as file:
    agent = pickle.load(file)
policy = agent.policy

trajectories = rollout_policy(environment, policy, num_episodes=NUM_EPISODES)

rewards = []
for trajectory in trajectories:
    reward = 0
    for observation in trajectory:
        reward += observation.reward
    rewards.append(reward)

plt.plot(agent.episodes_cumulative_rewards, label='Agent Learning')
plt.plot(rewards, label='Executed Policy')

plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.legend(loc='best')
plt.show()
