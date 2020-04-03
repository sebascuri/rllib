import numpy as np
import torch.nn.functional as func
import torch.optim

from rllib.util.training import train_agent, evaluate_agent
from rllib.agent import SoftQLearningAgent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.value_function import NNQFunction

# ENVIRONMENT = 'NChain-v0'
ENVIRONMENT = 'CartPole-v0'

NUM_EPISODES = 50
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
MOMENTUM = 0.1
WEIGHT_DECAY = 1e-4
GAMMA = 0.99
TEMPERATURE = 0.1
LAYERS = [64, 64]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)

optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

agent = SoftQLearningAgent(
    environment.name, q_function, criterion, optimizer, memory, batch_size=BATCH_SIZE,
    temperature=TEMPERATURE, target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, environment, 1, MAX_STEPS)
