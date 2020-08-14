import pytest
import torch
import torch.testing

from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNQFunction


@pytest.fixture(params=[0.0, 0.9])
def eps_start(request):
    return request.param


@pytest.fixture
def q_function():
    return NNQFunction(num_actions=2, num_states=4, dim_state=(), dim_action=())


def test_epsilon(eps_start, q_function):
    policy = EpsGreedy(q_function, ExponentialDecay(eps_start))
    for t in range(100):
        assert policy.param() == eps_start
        policy.update()

    policy = EpsGreedy(
        q_function, ExponentialDecay(start=eps_start, end=0.1, decay=100)
    )
    for t in range(100):
        torch.testing.assert_allclose(
            policy.param(), 0.1 + (eps_start - 0.1) * torch.exp(-torch.tensor(t / 100))
        )
        policy.update()


def test_discrete(eps_start, q_function):
    policy = EpsGreedy(q_function, eps_start)
    for _ in range(100):
        state = torch.randint(4, ())
        action = q_function(state).argmax(dim=-1)
        probs = eps_start / 2 * torch.ones(2)
        probs[action] += 1 - eps_start

        assert (tensor_to_distribution(policy(state)).probs == probs).all()
