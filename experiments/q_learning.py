import matplotlib.pyplot as plt
from rllib.agent import QLearningAgent, GradientQLearningAgent, \
    DeepQLearningAgent, DoubleDQNAgent
from rllib.util import rollout_agent
from rllib.util.neural_networks import DeterministicNN
from rllib.dataset import ExperienceReplay
from rllib.dataset.transforms import StateNormalizer
from rllib.exploration_strategies import EpsGreedy
# from rllib.environment.systems import InvertedPendulum, CartPole
from rllib.environment import GymEnvironment
import numpy as np
import torch.nn.functional as func
import torch.optim

ENVIRONMENT = 'CartPole-v0'
NUM_EPISODES = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.9
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0

for Agent in [QLearningAgent, GradientQLearningAgent, DeepQLearningAgent,
              DoubleDQNAgent]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(ENVIRONMENT, SEED)
    exploration = EpsGreedy(EPS_START, EPS_END, EPS_DECAY)
    q_function = DeterministicNN(environment.dim_state, environment.num_action,
                                 layers=LAYERS)
    q_target = DeterministicNN(environment.dim_state, environment.num_action,
                               layers=LAYERS)

    optimizer = torch.optim.Adam
    criterion = func.mse_loss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, transforms=[StateNormalizer()])

    hyper_params = {
        'target_update_frequency': TARGET_UPDATE_FREQUENCY,
        'target_update_tau': TARGET_UPDATE_TAU,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE
    }
    agent = Agent(q_function, q_target, exploration, criterion, optimizer, memory,
                  hyper_params)
    rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

    plt.plot(agent.episodes_steps, label=str(agent))
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.legend(loc='best')
plt.show()
