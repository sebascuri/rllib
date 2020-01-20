"""Mountain Car Solution."""
import matplotlib.pyplot as plt
from rllib.util import rollout_agent
from rllib.environment import GymEnvironment
from rllib.policy import FelixPolicy
from rllib.value_function import NNQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import GaussianExploration
from rllib.agent import DDPGAgent
import torch.nn.functional as func
import numpy as np
import torch.optim
import pickle

ENVIRONMENT = 'MountainCarContinuous-v0'
NUM_EPISODES = 25
MILESTONES = [0, 5, NUM_EPISODES - 1]
MAX_STEPS = 2500
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 25000
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
# MOMENTUM = 0.1
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
policy = FelixPolicy(environment.dim_state, environment.dim_action, temperature=0.)
exploration = GaussianExploration(EPS_START, EPS_END, EPS_DECAY)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)
actor_optimizer = torch.optim.Adam(policy.parameters, lr=ACTOR_LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
critic_optimizer = torch.optim.Adam(q_function.parameters, lr=CRITIC_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
criterion = func.mse_loss

agent = DDPGAgent(q_function, policy, exploration, criterion, critic_optimizer,
                  actor_optimizer, memory,
                  target_update_frequency=TARGET_UPDATE_FREQUENCY,
                  gamma=GAMMA,
                  episode_length=MAX_STEPS)

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES,
              milestones=MILESTONES, render=RENDER)

with open('{}_{}.pkl'.format(environment.name, agent.name), 'wb') as file:
    pickle.dump(agent, file)

plt.plot(agent.episodes_cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('{} in {}'.format(agent.__class__.__name__, ENVIRONMENT))
plt.show()
