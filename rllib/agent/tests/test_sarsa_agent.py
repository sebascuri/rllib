import pytest
from rllib.agent import SARSAAgent, ExpectedSARSAAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction, TabularQFunction
from rllib.policy import EpsGreedy, SoftMax, MellowMax
from rllib.environment import GymEnvironment, EasyGridWorld
import torch.optim

NUM_EPISODES = 10
MAX_STEPS = 25
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.9
MEMORY_MAX_SIZE = 5000
LEARNING_RATE = 0.001
BATCH_SIZE = 4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0


@pytest.fixture(params=['CartPole-v0', 'NChain-v0'])
def environment(request):
    return request.param


@pytest.fixture(params=[SARSAAgent, ExpectedSARSAAgent])
def agent(request):
    return request.param


@pytest.fixture(params=[SoftMax, MellowMax, EpsGreedy])
def policy(request):
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request):
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

    optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    q_agent = agent(
        q_function=q_function, policy=policy,
        criterion=criterion, optimizer=optimizer,
        target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA,
        exploration_episodes=2)
    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)


def test_policies(environment, policy, batch_size):
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
    q_agent = SARSAAgent(q_function=q_function, policy=policy,
                         criterion=criterion, optimizer=optimizer,
                         batch_size=batch_size,
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

    q_agent = agent(q_function=q_function, policy=policy,
                    criterion=criterion, optimizer=optimizer,
                    target_update_frequency=TARGET_UPDATE_FREQUENCY, gamma=GAMMA)

    rollout_agent(environment, q_agent, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES)
    print(q_function.table)
