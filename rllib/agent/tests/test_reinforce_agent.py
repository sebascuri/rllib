from rllib.agent import REINFORCE
from rllib.environment import GymEnvironment
from rllib.util.rollout import rollout_agent
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction
import torch
import numpy as np
import torch.nn.functional as func
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
    policy_optimizer = torch.optim.Adam(policy.parameters, lr=ACTOR_LEARNING_RATE)

    if baseline:
        value_function = NNValueFunction(environment.dim_state,
                                         num_states=environment.num_states,
                                         layers=LAYERS)
        value_optimizer = torch.optim.Adam(value_function.parameters,
                                           lr=CRITIC_LEARNING_RATE)
    else:
        value_function, value_optimizer = None, None

    criterion = func.mse_loss

    agent = REINFORCE(policy=policy, policy_optimizer=policy_optimizer,
                      baseline=value_function, baseline_optimizer=value_optimizer,
                      criterion=criterion, num_rollouts=num_rollouts, gamma=GAMMA)

    rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
