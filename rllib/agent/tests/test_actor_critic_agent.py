import numpy as np
import pytest
import torch

from rllib.agent import ActorCriticAgent, A2CAgent, GAACAgent, ExpectedActorCriticAgent
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction, NNQFunction

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


@pytest.fixture(params=[ActorCriticAgent, A2CAgent, ExpectedActorCriticAgent])
def agent(request):
    return request.param


def test_ac_agent(environment, agent, num_rollouts):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    policy = NNPolicy(environment.dim_state, environment.dim_action,
                      num_states=environment.num_states,
                      num_actions=environment.num_actions,
                      layers=LAYERS)

    critic = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions, layers=LAYERS)
    optimizer = torch.optim.Adam([
        {'params': critic.parameters(), 'lr': CRITIC_LEARNING_RATE},
        {'params': policy.parameters(), 'lr': ACTOR_LEARNING_RATE},
    ])

    criterion = torch.nn.MSELoss

    agent = agent(environment.name, policy=policy, critic=critic, optimizer=optimizer,
                  criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)


def test_gaac_agent(environment, num_rollouts):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    policy = NNPolicy(environment.dim_state, environment.dim_action,
                      num_states=environment.num_states,
                      num_actions=environment.num_actions,
                      layers=LAYERS)

    critic = NNValueFunction(environment.dim_state, num_states=environment.num_states,
                             layers=LAYERS)
    optimizer = torch.optim.Adam([
        {'params': critic.parameters(), 'lr': CRITIC_LEARNING_RATE},
        {'params': policy.parameters(), 'lr': ACTOR_LEARNING_RATE},
    ])
    criterion = torch.nn.MSELoss

    agent = GAACAgent(environment.name, policy=policy,
                      critic=critic, optimizer=optimizer,
                      criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
