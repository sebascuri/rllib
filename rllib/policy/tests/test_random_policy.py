from rllib.policy import RandomPolicy
import torch
from torch.distributions import Normal, Categorical
import pytest


@pytest.fixture(params=[(True, 4, 2), (False, 4, 4)])
def random_policy(request):
    discrete = request.param[0]
    state_dim = request.param[1]
    action_dim = request.param[2]
    if discrete:
        action_space = Categorical(torch.ones((action_dim,)))
    else:
        action_space = Normal(torch.zeros((action_dim,)), torch.ones((action_dim,)))

    return RandomPolicy(action_space, state_dim), state_dim, action_dim


def test_init(random_policy):
    pass


def test_random_action(random_policy):
    nn_policy, state_dim, action_dim = random_policy
    distribution = nn_policy.random_action()
    sample = distribution.sample()
    if distribution.has_enumerate_support:  # Discrete
        assert distribution.logits.shape == (action_dim,)
        assert sample.shape == ()
    else:  # Continuous
        assert distribution.mean.shape == (action_dim,)
        assert sample.shape == (action_dim,)


def test_forward(random_policy):
    nn_policy, state_dim, action_dim = random_policy
    state = torch.randn(state_dim, )
    distribution = nn_policy.action(state)
    sample = distribution.sample()

    if distribution.has_enumerate_support:  # Discrete
        assert distribution.logits.shape == (action_dim,)
        assert sample.shape == ()
    else:  # Continuous
        assert distribution.mean.shape == (action_dim,)
        assert sample.shape == (action_dim,)
