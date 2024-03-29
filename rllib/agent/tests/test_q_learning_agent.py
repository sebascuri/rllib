import pytest
import torch
import torch.optim
import torch.testing

from rllib.agent import DDQNAgent, DQNAgent, QLearningAgent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.environment.mdps import EasyGridWorld
from rllib.policy import EpsGreedy, MellowMax, SoftMax
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction, TabularQFunction

NUM_EPISODES = 10
MAX_STEPS = 25
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.1
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0


@pytest.fixture(params=["CartPole-v0", "NChain-v0"])
def environment(request):
    return request.param


@pytest.fixture(params=[QLearningAgent, DQNAgent, DDQNAgent])
def agent(request):
    return request.param


@pytest.fixture(params=[EpsGreedy, SoftMax, MellowMax])
def policy(request):
    return request.param


def test_policies(environment, policy):
    environment = GymEnvironment(environment, SEED)

    critic = NNQFunction(
        dim_state=environment.dim_observation,
        dim_action=environment.dim_action,
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        layers=LAYERS,
        tau=TARGET_UPDATE_TAU,
    )

    policy = policy(critic, 0.1)

    optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

    agent = DDQNAgent(
        critic=critic,
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        memory=memory,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        gamma=GAMMA,
    )
    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
    agent.logger.delete_directory()  # Cleanup directory.


def test_tabular_interaction(agent, policy):
    LEARNING_RATE = 0.1
    environment = EasyGridWorld()

    critic = TabularQFunction(
        num_states=environment.num_states, num_actions=environment.num_actions
    )
    policy = policy(critic, 0.1)
    optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

    agent = agent(
        critic=critic,
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        memory=memory,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        gamma=GAMMA,
    )

    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
    agent.logger.delete_directory()  # Cleanup directory.

    torch.testing.assert_allclose(
        critic.table.shape,
        torch.Size([environment.num_actions, environment.num_states, 1]),
    )
