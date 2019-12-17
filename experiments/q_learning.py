import matplotlib.pyplot as plt
from rllib.agent import QLearningAgent, GQLearningAgent, DQNAgent, DDQNAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import EpsGreedy
# from rllib.environment.systems import InvertedPendulum, CartPole
from rllib.environment import GymEnvironment
import numpy as np
import torch.nn.functional as func
import torch.optim
import pickle


# ENVIRONMENT = 'NChain-v0'
ENVIRONMENT = 'CartPole-v0'

NUM_EPISODES = 100
MAX_STEPS = 200
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

for name, Agent in {'DDQN': DDQNAgent,
                    'Q-Learning': QLearningAgent,
                    'GQ-Learning': GQLearningAgent,
                    'DQN': DQNAgent}.items():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(ENVIRONMENT, SEED, MAX_STEPS)
    exploration = EpsGreedy(EPS_START, EPS_END, EPS_DECAY)
    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_states,
                             num_actions=environment.num_actions,
                             layers=LAYERS,
                             tau=TARGET_UPDATE_TAU
                             )

    optimizer = torch.optim.Adam
    criterion = func.mse_loss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

    hyper_params = {
        'target_update_frequency': TARGET_UPDATE_FREQUENCY,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE
    }
    agent = Agent(q_function, exploration, criterion, optimizer, memory, hyper_params,
                  episode_length=MAX_STEPS)
    rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

    plt.plot(agent.episodes_cumulative_rewards, label=name)
    # with open('../runs/{}_{}.pkl'.format(ENVIRONMENT, name), 'wb') as file:
    #     pickle.dump(agent, file)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.show()
