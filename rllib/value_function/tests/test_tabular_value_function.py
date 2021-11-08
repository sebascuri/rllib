import pytest
import torch
import torch.testing

from rllib.util.neural_networks.utilities import random_tensor
from rllib.value_function import TabularQFunction, TabularValueFunction


@pytest.fixture(params=[4, 2, 1])
def num_states(request):
    return request.param


@pytest.fixture(params=[4, 2, 1])
def num_actions(request):
    return request.param


@pytest.fixture(params=[4, 1])
def dim_reward(request):
    return request.param


@pytest.fixture(params=[None, 1, 16])
def batch_size(request):
    return request.param


class TestTabularValueFunction(object):
    def test_init(self, dim_reward):
        value_function = TabularValueFunction(num_states=4, dim_reward=(dim_reward,))
        torch.testing.assert_allclose(
            value_function.table, torch.zeros(1, 4, dim_reward)
        )

    def test_set_value(self, dim_reward):
        value_function = TabularValueFunction(num_states=4, dim_reward=(dim_reward,))
        value_function.set_value(2, torch.ones(dim_reward))
        table = torch.zeros(1, 4, dim_reward)
        table[:, 2, :] = 1.0
        torch.testing.assert_allclose(value_function.table, table)

    def test_forward(self, num_states, batch_size, dim_reward):
        value_function = TabularValueFunction(
            num_states=num_states, dim_reward=(dim_reward,)
        )
        state = random_tensor(True, num_states, batch_size)
        value = value_function(state)

        assert value.shape == torch.Size(
            [batch_size, dim_reward] if batch_size else [dim_reward]
        )
        assert value.dtype is torch.get_default_dtype()


class TestTabularQFunction(object):
    def test_init(self, dim_reward):
        value_function = TabularQFunction(
            num_states=4, num_actions=2, dim_reward=(dim_reward,)
        )
        torch.testing.assert_allclose(
            value_function.table, torch.zeros(2, 4, dim_reward)
        )

    def test_set_value(self, dim_reward):
        value_function = TabularQFunction(
            num_states=4, num_actions=2, dim_reward=(dim_reward,)
        )
        value_function.set_value(2, 1, torch.tensor([1.0]))
        table = torch.zeros(2, 4, dim_reward)
        table[1, 2, :] = 1.0 * torch.ones(dim_reward)
        torch.testing.assert_allclose(value_function.table, table)

    def test_forward(self, num_states, num_actions, batch_size, dim_reward):
        q_function = TabularQFunction(
            num_states=num_states, num_actions=num_actions, dim_reward=(dim_reward,)
        )

        state = random_tensor(True, num_states, batch_size)
        action = random_tensor(True, num_actions, batch_size)
        value = q_function(state, action)
        assert value.shape == torch.Size(
            [batch_size, dim_reward] if batch_size else [dim_reward]
        )
        assert value.dtype is torch.get_default_dtype()

    def test_partial_q_function(self, num_states, num_actions, batch_size, dim_reward):
        q_function = TabularQFunction(
            num_states=num_states, num_actions=num_actions, dim_reward=(dim_reward,)
        )
        state = random_tensor(True, num_states, batch_size)

        action_value = q_function(state)
        assert action_value.shape == torch.Size(
            [batch_size, num_actions, dim_reward]
            if batch_size
            else [num_actions, dim_reward]
        )
        assert action_value.dtype is torch.get_default_dtype()
