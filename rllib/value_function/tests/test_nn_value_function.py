import pytest
import torch
import torch.testing

from rllib.util.neural_networks import random_tensor
from rllib.value_function import NNValueFunction, NNQFunction
from rllib.value_function import TabularValueFunction, TabularQFunction


@pytest.fixture(params=[False, True])
def discrete_state(request):
    return request.param


@pytest.fixture(params=[False, True])
def discrete_action(request):
    return request.param


@pytest.fixture(params=[4, 2, 1])
def dim_state(request):
    return request.param


@pytest.fixture(params=[4, 2, 1])
def dim_action(request):
    return request.param


@pytest.fixture(params=[None, 1, 16])
def batch_size(request):
    return request.param


class TestNNValueFunction(object):
    def init(self, discrete_state, dim_state):
        if discrete_state:
            self.num_states = dim_state
            self.dim_state = 1
        else:
            self.num_states = None
            self.dim_state = dim_state

        layers = [32, 32]
        self.value_function = NNValueFunction(self.dim_state, self.num_states, layers)

    def test_num_states(self, discrete_state, dim_state):
        self.init(discrete_state, dim_state)
        assert (self.num_states if self.num_states is not None else -1) == self.value_function.num_states
        assert discrete_state == self.value_function.discrete_state

    def test_dim_states(self, discrete_state, dim_state):
        self.init(discrete_state, dim_state)
        assert self.dim_state == self.value_function.dim_state

    def test_forward(self, discrete_state, dim_state, batch_size):
        self.init(discrete_state, dim_state)
        state = random_tensor(discrete_state, dim_state, batch_size)
        value = self.value_function(state)

        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()


class TestNNQFunction(object):
    def init(self, discrete_state, discrete_action, dim_state, dim_action):
        if discrete_state:
            self.num_states = dim_state
            self.dim_state = 1
        else:
            self.num_states = None
            self.dim_state = dim_state

        if discrete_action:
            self.num_actions = dim_action
            self.dim_action = 1
        else:
            self.num_actions = None
            self.dim_action = dim_action

        layers = [32, 32]
        self.q_function = NNQFunction(self.dim_state, self.dim_action,
                                      self.num_states, self.num_actions, layers)

    def test_init(self, discrete_state, discrete_action, dim_state, dim_action):
        if discrete_state and not discrete_action:
            with pytest.raises(NotImplementedError):
                self.init(discrete_state, discrete_action, dim_state, dim_action)

    def test_num_states_actions(self, discrete_state, discrete_action, dim_state, dim_action):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            assert (self.num_states if self.num_states is not None else -1) == self.q_function.num_states
            assert (self.num_actions if self.num_actions is not None else -1) == self.q_function.num_actions

            assert discrete_state == self.q_function.discrete_state
            assert discrete_action == self.q_function.discrete_action

    def test_dim_state_actions(self, discrete_state, discrete_action, dim_state, dim_action):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            assert self.dim_state == self.q_function.dim_state
            assert self.dim_action == self.q_function.dim_action

    def test_forward(self, discrete_state, discrete_action, dim_state, dim_action, batch_size):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            state = random_tensor(discrete_state, dim_state, batch_size)
            action = random_tensor(discrete_action, dim_action, batch_size)
            print(state.shape, action.shape)
            value = self.q_function(state, action)
            assert value.shape == torch.Size([batch_size] if batch_size else [])
            assert value.dtype is torch.get_default_dtype()

    def test_partial_q_function(self, discrete_state, discrete_action, dim_state, dim_action, batch_size):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            state = random_tensor(discrete_state, dim_state, batch_size)

            if not discrete_action:
                with pytest.raises(NotImplementedError):
                    self.q_function(state)
            else:
                action_value = self.q_function(state)
                assert action_value.shape == torch.Size(
                    [batch_size, self.num_actions] if batch_size else [self.num_actions]
                )
                assert action_value.dtype is torch.get_default_dtype()


class TestTabularValueFunction(object):
    def test_init(self):
        value_function = TabularValueFunction(num_states=4)
        torch.testing.assert_allclose(value_function.table, 0)

    def test_set_value(self):
        value_function = TabularValueFunction(num_states=4)
        value_function.set_value(2, 1.)
        torch.testing.assert_allclose(value_function.table, torch.tensor([0, 0, 1., 0]))


class TestTabularQFunction(object):
    def test_init(self):
        value_function = TabularQFunction(num_states=4, num_actions=2)
        torch.testing.assert_allclose(value_function.table, 0)

    def test_set_value(self):
        value_function = TabularQFunction(num_states=4, num_actions=2)
        value_function.set_value(2, 1, 1.)
        torch.testing.assert_allclose(value_function.table, torch.tensor([[0, 0, 0., 0],
                                                                         [0, 0, 1., 0]])
                                      )
