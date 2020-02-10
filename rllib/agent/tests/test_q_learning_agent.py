import pytest
from rllib.agent import DQNAgent, QLearningAgent, GQLearningAgent, DDQNAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction, TabularQFunction
from rllib.dataset import ExperienceReplay
from rllib.policy import EpsGreedy
from rllib.environment import GymEnvironment, EasyGridWorld
import torch.nn.functional as func
import torch.optim

NUM_EPISODES = 25
MAX_STEPS = 25
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


@pytest.fixture(params=['CartPole-v0', 'NChain-v0'])
def environment(request):
    return request.param


@pytest.fixture(params=[DQNAgent, QLearningAgent, GQLearningAgent, DDQNAgent])
def agent(request):
    return request.param


def test_nnq_interaction(environment, agent):
    environment = GymEnvironment(environment, SEED)

    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_states,
                             num_actions=environment.num_actions,
                             layers=LAYERS,
                             tau=TARGET_UPDATE_TAU,
                             )
    policy = EpsGreedy(q_function, EPS_START, EPS_END, EPS_DECAY)

    optimizer = torch.optim.Adam(q_function.parameters, lr=LEARNING_RATE)
    criterion = func.mse_loss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

    q_agent = agent(q_function=q_function, policy=policy,
                    criterion=criterion, optimizer=optimizer, memory=memory,
                    target_update_frequency=TARGET_UPDATE_FREQUENCY,
                    episode_length=MAX_STEPS, gamma=GAMMA)
    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)


def test_tabular_interaction(agent):
    LEARNING_RATE = 0.1
    environment = EasyGridWorld()

    q_function = TabularQFunction(num_states=environment.num_states,
                                  num_actions=environment.num_actions)
    policy = EpsGreedy(q_function, EPS_START, EPS_END, EPS_DECAY)
    optimizer = torch.optim.Adam(q_function.parameters, lr=LEARNING_RATE)
    criterion = func.mse_loss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

    q_agent = agent(q_function=q_function, policy=policy,
                    criterion=criterion, optimizer=optimizer, memory=memory,
                    target_update_frequency=TARGET_UPDATE_FREQUENCY,
                    episode_length=10, gamma=GAMMA)

    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)
    print(q_function.table)
