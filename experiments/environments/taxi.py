"""Python Script Template."""
import matplotlib.pyplot as plt
from rllib.agent import GQLearningAgent
from rllib.environment import GymEnvironment
from rllib.value_function import NNQFunction
from rllib.dataset import ExperienceReplay
from rllib.policy import EpsGreedy
from rllib.util import rollout_agent
import torch
import torch.nn.functional as func
import pickle

ENVIRONMENT = 'Taxi-v2'
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
q_function.q_function.head.weight.data = torch.ones_like(
    q_function.q_function.head.weight)

policy = EpsGreedy(q_function, EPSILON)
optimizer = torch.optim.SGD(q_function.parameters, lr=LEARNING_RATE,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

agent = GQLearningAgent(q_function, policy, criterion, optimizer, memory,
                        target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)
rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES,
              milestones=MILESTONES)

plt.plot(agent.episodes_cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()

with open('{}_{}.pkl'.format(environment.name, agent.name), 'wb') as file:
    pickle.dump(agent, file)

rollout_agent(environment, agent, num_episodes=1, render=True)
