import matplotlib.pyplot as plt
from rllib.util import rollout_agent, rollout_policy
from rllib.environment.systems import InvertedPendulum, GaussianSystem
from rllib.environment import SystemEnvironment, GymEnvironment
from rllib.policy import FelixPolicy
from rllib.value_function import NNQFunction, TabularQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import GaussianNoise
from rllib.agent import DDPGAgent
import torch.nn.functional as func
import numpy as np
import torch.optim
import pickle

ENVIRONMENT = ['MountainCarContinuous-v0', 'Pendulum-v0'][0]
NUM_EPISODES = 25
MAX_STEPS = 2500
TARGET_UPDATE_FREQUENCY = 2
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
LAYERS = [64, 64]
SEED = 0
RENDER = True

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = FelixPolicy(environment.dim_state, environment.dim_action, deterministic=True)
noise = GaussianNoise(EPS_START, EPS_END, EPS_DECAY)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)
actor_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
critic_optimizer = torch.optim.Adam(q_function.parameters(), lr=CRITIC_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss

agent = DDPGAgent(q_function, policy, noise, criterion, critic_optimizer,
                  actor_optimizer, memory,
                  target_update_frequency=TARGET_UPDATE_FREQUENCY,
                  gamma=GAMMA)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
plt.plot(agent.episodes_cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)

# rollout_policy(environment, agent.policy, max_steps=100, render=True)
