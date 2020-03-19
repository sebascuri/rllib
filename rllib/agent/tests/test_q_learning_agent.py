import pytest
import torch.nn.functional as func
import torch.optim

from rllib.agent import QLearningAgent, DQNAgent, DDQNAgent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment, EasyGridWorld
from rllib.policy import EpsGreedy, SoftMax, MellowMax
from rllib.util import rollout_agent
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction, TabularQFunction

NUM_EPISODES = 10
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


@pytest.fixture(params=[QLearningAgent, DQNAgent, DDQNAgent])
def agent(request):
    return request.param


@pytest.fixture(params=[EpsGreedy, SoftMax, MellowMax])
def policy(request):
    return request.param


def test_nnq_interaction(environment, agent):
    environment = GymEnvironment(environment, SEED)

    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_states,
                             num_actions=environment.num_actions,
                             layers=LAYERS,
                             tau=TARGET_UPDATE_TAU,
                             )
    policy = EpsGreedy(q_function, ExponentialDecay(EPS_START, EPS_END, EPS_DECAY))

    optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

    q_agent = agent(q_function=q_function, policy=policy,
                    criterion=criterion, optimizer=optimizer, memory=memory,
                    target_update_frequency=TARGET_UPDATE_FREQUENCY,
                    gamma=GAMMA,
                    exploration_steps=2)
    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)


def test_policies(environment, policy):
    environment = GymEnvironment(environment, SEED)

    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_states,
                             num_actions=environment.num_actions,
                             layers=LAYERS,
                             tau=TARGET_UPDATE_TAU,
                             )

    policy = policy(q_function, 0.1)

    optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

    q_agent = DDQNAgent(
        q_function=q_function, policy=policy,
        criterion=criterion, optimizer=optimizer, memory=memory,
        target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)
    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)


def test_tabular_interaction(agent, policy):
    LEARNING_RATE = 0.1
    environment = EasyGridWorld()

    q_function = TabularQFunction(num_states=environment.num_states,
                                  num_actions=environment.num_actions)
    policy = policy(q_function, 0.1)
    optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)

    q_agent = agent(
        q_function=q_function, policy=policy,
        criterion=criterion, optimizer=optimizer, memory=memory,
        target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)

    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)
    print(q_function.table)
