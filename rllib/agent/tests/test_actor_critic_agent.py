from rllib.agent import ActorCriticAgent, A2CAgent, GAACAgent
from rllib.environment import GymEnvironment
from rllib.util.rollout import rollout_agent
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction, NNQFunction
import torch
import numpy as np
import pytest

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


@pytest.fixture(params=[ActorCriticAgent, A2CAgent])
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
    critic_optimizer = torch.optim.Adam(critic.parameters(),
                                        lr=CRITIC_LEARNING_RATE)

    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)
    criterion = torch.nn.MSELoss

    agent = agent(policy=policy, actor_optimizer=actor_optimizer,
                  critic=critic, critic_optimizer=critic_optimizer,
                  criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

    rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)


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
    critic_optimizer = torch.optim.Adam(critic.parameters(),
                                        lr=CRITIC_LEARNING_RATE)

    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)
    criterion = torch.nn.MSELoss

    agent = GAACAgent(policy=policy, actor_optimizer=actor_optimizer,
                      critic=critic, critic_optimizer=critic_optimizer,
                      criterion=criterion, num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

    rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
