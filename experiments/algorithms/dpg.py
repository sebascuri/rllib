import numpy as np
import torch.nn.functional as func
import torch.optim

from experiments.util import train, evaluate
from rllib.agent import TD3Agent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.exploration_strategies import GaussianNoise
from rllib.policy import FelixPolicy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

ENVIRONMENT = ['MountainCarContinuous-v0', 'Pendulum-v0'][0]
NUM_EPISODES = 25
MAX_STEPS = 2500
TARGET_UPDATE_FREQUENCY = 2
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 25000
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e6
LAYERS = [64, 64]
SEED = 0
RENDER = True

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = FelixPolicy(environment.dim_state, environment.dim_action, deterministic=True,
                     tau=TARGET_UPDATE_TAU)
noise = GaussianNoise(ExponentialDecay(EPS_START, EPS_END, EPS_DECAY))
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

agent = TD3Agent(
    q_function, policy, noise, criterion, critic_optimizer,
    actor_optimizer, memory,
    target_update_frequency=TARGET_UPDATE_FREQUENCY,
    gamma=GAMMA)

train(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate(agent, environment, 1, MAX_STEPS)
