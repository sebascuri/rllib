from rllib.policy import RandomPolicy
import torch
# from torch.distributions import Normal, Categorical
import pytest


@pytest.fixture(params=[(4, 2, None), (4, 1, 4)])
def random_policy(request):
    dim_state = request.param[0]
    dim_action = request.param[1]
    num_action = request.param[2]

    return (RandomPolicy(dim_state, dim_action, num_action), dim_state, dim_action,
            num_action)


def test_init(random_policy):
    pass


def test_random_action(random_policy):
    policy, dim_state, dim_action, num_action = random_policy
    distribution = policy.random()
    sample = distribution.sample()
    if distribution.has_enumerate_support:  # Discrete
        assert distribution.logits.shape == (num_action,)
        assert sample.shape == ()
    else:  # Continuousgit st
        assert distribution.mean.shape == (dim_action,)
        assert sample.shape == (dim_action,)


def test_forward(random_policy):
    policy, dim_state, dim_action, num_action = random_policy
    state = torch.randn(dim_state, )
    distribution = policy.action(state)
    sample = distribution.sample()

    if distribution.has_enumerate_support:  # Discrete
        assert distribution.logits.shape == (num_action,)
        assert sample.shape == ()
    else:  # Continuous
        assert distribution.mean.shape == (dim_action,)
        assert sample.shape == (dim_action,)
