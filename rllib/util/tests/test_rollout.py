from rllib.util import rollout_policy, rollout_agent
from rllib.policy import RandomPolicy
from rllib.agent import RandomAgent
import torch
from torch.distributions import Categorical, Normal
import pytest
import gym


@pytest.fixture(params=['CartPole-v0', 'Pendulum-v0'])
def environment(request):
    env = gym.make(request.param)
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        action_space = Categorical(torch.ones(action_dim))
        action_dim = 1
    else:
        action_dim = env.action_space.shape[0]
        action_space = Normal(torch.zeros(action_dim), torch.ones(action_dim))

    return env, action_space, state_dim, action_dim


def test_rollout_policy(environment):
    environment, action_space, state_dim, action_dim = environment
    policy = RandomPolicy(action_space, state_dim)
    trajectory = rollout_policy(environment, policy)

    assert len(trajectory) > 0


def test_rollout_agent(environment):
    environment, action_space, state_dim, action_dim = environment
    agent = RandomAgent(state_dim, action_dim, action_space)
    rollout_agent(environment, agent)
