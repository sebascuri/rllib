import pytest
import torch
import torch.testing

from rllib.policy import SoftMax
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import NNQFunction


@pytest.fixture(params=[0.1, 1.0])
def t_start(request):
    return request.param


@pytest.fixture
def q_function():
    return NNQFunction(num_actions=2, dim_action=(), num_states=4, dim_state=())


def test_discrete(t_start, q_function):
    policy = SoftMax(q_function, t_start)
    for _ in range(100):
        state = torch.randint(4, ())
        logits = q_function(state)
        probs = torch.softmax(logits / t_start, dim=0)
        torch.testing.assert_allclose(
            tensor_to_distribution(policy(state)).probs, probs
        )
