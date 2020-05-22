import numpy as np
import pytest
import torch

from rllib.agent import REINFORCEAgent
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction

MAX_STEPS = 25
NUM_EPISODES = 25
TARGET_UPDATE_FREQUENCY = 1
NUM_ROLLOUTS = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-2

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0


@pytest.fixture(params=['CartPole-v0', 'NChain-v0'])
def environment(request):
    return request.param


@pytest.fixture(params=[1, 4])
def num_rollouts(request):
    return request.param


@pytest.fixture(params=[True, False])
def baseline(request):
    return request.param


def test_REINFORCE(environment, num_rollouts, baseline):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    policy = NNPolicy(environment.dim_state, environment.dim_action,
                      num_states=environment.num_states,
                      num_actions=environment.num_actions,
                      layers=LAYERS)

    if baseline:
        value_function = NNValueFunction(environment.dim_state,
                                         num_states=environment.num_states,
                                         layers=LAYERS)
        optimizer = torch.optim.Adam([
            {'params': policy.parameters(), 'lr': ACTOR_LEARNING_RATE},
            {'params': value_function.parameters(), 'lr': CRITIC_LEARNING_RATE}
        ])

    else:
        value_function = None
        optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)

    criterion = torch.nn.MSELoss

    agent = REINFORCEAgent(environment.name, policy=policy, baseline=value_function,
                           optimizer=optimizer, criterion=criterion,
                           num_rollouts=num_rollouts, gamma=GAMMA)

    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
