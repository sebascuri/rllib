from rllib.policy import EpsGreedy
from rllib.value_function import NNQFunction
from rllib.dataset.datatypes import Observation
import torch
import torch.testing
import pytest
import numpy as np


@pytest.fixture(params=[0.0, 0.9])
def eps_start(request):
    return request.param


@pytest.fixture
def q_function():
    return NNQFunction(num_actions=2, num_states=4, dim_state=1, dim_action=1)


def test_epsilon(eps_start, q_function):
    policy = EpsGreedy(q_function, start=eps_start)
    for t in range(100):
        assert policy.param() == eps_start
        policy.update(Observation(1, 2, 3, 4, True))

    policy = EpsGreedy(q_function, start=eps_start, end=0.1, decay=100)
    for t in range(100):
        assert policy.param() == 0.1 + (eps_start - 0.1) * np.exp(-t / 100)
        policy.update(Observation(1, 2, 3, 4, True))


def test_discrete(eps_start, q_function):
    policy = EpsGreedy(q_function, start=eps_start)
    for t in range(100):
        state = torch.randint(4, ())
        action = q_function(state).argmax(dim=-1)
        probs = eps_start / 2 * torch.ones(2)
        probs[action] += (1 - eps_start)

        assert (policy(state).probs == probs).all()
