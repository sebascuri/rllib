import matplotlib.pyplot as plt
from rllib.agent import DDQNAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import EpsGreedy, BoltzmannExploration
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

eps_greedy = EpsGreedy(eps_start=1., eps_end=0.01, eps_decay=500)
boltzmann = BoltzmannExploration(t_start=1., t_end=0.01, t_decay=500)

for exploration in [eps_greedy, boltzmann]:

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(ENVIRONMENT, SEED)
    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_observations,
                             num_actions=environment.num_actions,
                             layers=LAYERS
                             )

    q_target = NNQFunction(environment.dim_observation, environment.dim_action,
                           num_states=environment.num_observations,
                           num_actions=environment.num_actions,
                           layers=LAYERS,
                           tau=TARGET_UPDATE_TAU
                           )

    optimizer = torch.optim.Adam
    criterion = func.mse_loss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

    hyper_params = {
        'target_update_frequency': TARGET_UPDATE_FREQUENCY,
        'target_update_tau': TARGET_UPDATE_TAU,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE
    }
    agent = DDQNAgent(q_function, q_target, exploration, criterion, optimizer, memory,
                      hyper_params)
    rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

    plt.plot(agent.episodes_cumulative_rewards, label=str(exploration))
plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.legend(loc='best')
plt.show()
