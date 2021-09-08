import pytest
import torch
import torch.testing

from rllib.util.neural_networks import random_tensor
from rllib.value_function import TabularQFunction, TabularValueFunction


@pytest.fixture(params=[4, 2, 1])
def num_states(request):
    return request.param


@pytest.fixture(params=[4, 2, 1])
def num_actions(request):
    return request.param


@pytest.fixture(params=[None, 1, 16])
def batch_size(request):
    return request.param


class TestTabularValueFunction(object):
    def test_init(self):
        value_function = TabularValueFunction(num_states=4)
        torch.testing.assert_allclose(value_function.table, torch.zeros(1, 4))

    def test_compile(self):
        torch.jit.script(TabularValueFunction(num_states=4))

    def test_set_value(self):
        value_function = TabularValueFunction(num_states=4)
        value_function.set_value(2, 1.0)
        torch.testing.assert_allclose(
            value_function.table, torch.tensor([[0, 0, 1.0, 0]])
        )

    def test_forward(self, num_states, batch_size):
        value_function = TabularValueFunction(num_states=num_states)
        state = random_tensor(True, num_states, batch_size)
        value = value_function(state)

        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()


class TestTabularQFunction(object):
    def test_init(self):
        value_function = TabularQFunction(num_states=4, num_actions=2)
        torch.testing.assert_allclose(value_function.table, torch.zeros(2, 4))

    def test_compile(self):
        torch.jit.script(TabularQFunction(num_states=4, num_actions=2))

    def test_set_value(self):
        value_function = TabularQFunction(num_states=4, num_actions=2)
        value_function.set_value(2, 1, 1.0)
        torch.testing.assert_allclose(
            value_function.table, torch.tensor([[0, 0, 0.0, 0], [0, 0, 1.0, 0]])
        )

    def test_forward(self, num_states, num_actions, batch_size):
        q_function = TabularQFunction(num_states=num_states, num_actions=num_actions)

        state = random_tensor(True, num_states, batch_size)
        action = random_tensor(True, num_actions, batch_size)
        value = q_function(state, action)
        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()

    def test_partial_q_function(self, num_states, num_actions, batch_size):
        q_function = TabularQFunction(num_states=num_states, num_actions=num_actions)
        state = random_tensor(True, num_states, batch_size)

        action_value = q_function(state)
        assert action_value.shape == torch.Size(
            [batch_size, num_actions] if batch_size else [num_actions]
        )
        assert action_value.dtype is torch.get_default_dtype()
