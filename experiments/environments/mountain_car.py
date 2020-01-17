"""Python Script Template."""
import matplotlib.pyplot as plt
from rllib.agent import QLearningAgent, GQLearningAgent, DQNAgent, DDQNAgent
from rllib.environment import GymEnvironment
from rllib.value_function import TabularQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import EpsGreedy
from torch.distributions import Categorical

from rllib.util import rollout_agent
import time

import numpy as np
import torch
import torch.nn.functional as func


ENVIRONMENT = 'Taxi-v2'
NUM_EPISODES = 10000
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 1
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
GAMMA = 0.6
EPSILON = 0.1
LAYERS = [64, 64]
SEED = 0
RENDER = True

environment = GymEnvironment(ENVIRONMENT, SEED)
q_table = np.ones([environment.num_states, environment.num_actions])
policy = np.ones([environment.num_states, environment.num_actions])

q_function = TabularQFunction(num_states=environment.num_states,
                              num_actions=environment.num_actions,
                              biased_head=False, tau=1)
for s in range(environment.num_states):
    for a in range(environment.num_actions):
        q_function.set_value(s, a, 1)

total_rewards = []
for i in range(NUM_EPISODES):
    state = environment.reset()
    done = False
    episode_rewards = 0
    while not done:
        state = torch.tensor(state)
        if np.random.rand() < EPSILON:
            # action = np.random.choice(environment.num_actions)
            action = torch.tensor(np.random.choice(environment.num_actions))
        else:
            action = q_function.argmax(state)
            # action = np.argmax(q_function(state))

        next_state, reward, done, _ = environment.step(action.item())
        tnext_state = torch.tensor(next_state)
        episode_rewards += reward

        td_error = reward + GAMMA * q_function.max(tnext_state) - q_function(state,
                                                                             action)
        # td_error = reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action]

        q_function.set_value(state, action,
                             q_function(state, action) + LEARNING_RATE * td_error)
        # q_table[state, action] += LEARNING_RATE * td_error
        state = next_state

    print(episode_rewards)
    total_rewards.append(episode_rewards)

plt.plot(total_rewards)
plt.show()

state = environment.reset()
done = False
r = 0
while not done:
    action = q_function.argmax(torch.tensor(state)).item()
    next_state, reward, done, _ = environment.step(action)
    state = next_state
    environment.render()
    time.sleep(1)
    r += reward
print(r)
# rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)
#
# fig, axes = plt.subplots(2, 1, sharex=False)
# axes[0].plot(agent.episodes_cumulative_rewards)
# tds = agent.logs['episode_td_errors']
# axes[1].plot(tds)
#
# axes[1].set_xlabel('Episode')
# axes[0].set_ylabel('Rewards')
# axes[0].legend(loc='best')
# axes[1].set_xlabel('Episode')
# axes[1].set_ylabel('Mean Absolute TD-Error')
# plt.show()
