import pytest
import torch
import torch.testing
from rllib.value_function import NNValueFunction, NNQFunction


class TestNNValueFunction(object):
    @pytest.fixture(params=[(4, None), (1, 4)], scope="class")
    def value_function(self, request):
        dim_state, num_states = request.param
        layers = [32, 32]
        return NNValueFunction(dim_state, num_states=num_states, layers=layers)

    def test_discrete_state(self, value_function):
        assert value_function.discrete_state or (value_function.num_states is None)

    def test_forward(self, value_function):
        for batch_size in [1, 16]:
            state = random_state(value_function, batch_size)
            value = value_function(state)
            assert value.shape == torch.Size([batch_size])
            assert value.dtype is torch.float

    def test_parameters(self, value_function):
        old_parameter = value_function.parameters
        value_function.parameters = old_parameter


class TestNNQFunction(object):
    @pytest.fixture(params=[(4, None, 2, None), (4, None, 1, 2), (1, 4, 1, 2)],
                    scope="class")
    def q_function(self, request):
        dim_state, num_states, dim_action, num_actions = request.param
        layers = [32, 32]
        return NNQFunction(dim_state, dim_action, num_states=num_states,
                           num_actions=num_actions, layers=layers)

    def test_discrete_action(self, q_function):
        assert q_function.discrete_state or (q_function.num_states is None)
        assert q_function.discrete_action or (q_function.num_actions is None)

    def test_init_exception(self):
        with pytest.raises(NotImplementedError):
            return NNQFunction(1, 2, num_states=4, num_actions=None)

    def test_forward(self, q_function):
        for batch_size in [16, 1]:
            state = random_state(q_function, batch_size)
            action = random_action(q_function, batch_size)
            value = q_function(state, action)
            assert value.shape == torch.Size([batch_size])
            assert value.dtype is torch.float

    def test_partial_q_function(self, q_function):
        for batch_size in [16, 1]:
            state = random_state(q_function, batch_size)

            if q_function.discrete_action:
                action_value = q_function(state)
                if batch_size == 1:
                    assert action_value.shape == torch.Size([q_function.num_actions])
                else:
                    assert action_value.shape == torch.Size([batch_size,
                                                             q_function.num_actions])
                assert action_value.dtype is torch.float
            else:
                with pytest.raises(NotImplementedError):
                    action_value = q_function(state)

    def test_parameters(self, q_function):
        old_parameter = q_function.parameters
        q_function.parameters = old_parameter

    def test_max(self, q_function):
        for batch_size in [16, 1]:
            state = random_state(q_function, batch_size)
            action = random_action(q_function, batch_size)
            if q_function.discrete_action:
                q_max = q_function.max(state)
                assert q_max.shape == action.shape
                assert q_max.dtype == torch.float
            else:
                with pytest.raises(NotImplementedError):
                    q_function.max(state)

    def test_argmax(self, q_function):
        for batch_size in [16, 1]:
            state = random_state(q_function, batch_size)
            action = random_action(q_function, batch_size)
            if q_function.discrete_action:
                best_action = q_function.argmax(state)
                assert best_action.shape == action.shape
                assert best_action.dtype is torch.long
            else:
                with pytest.raises(NotImplementedError):
                    q_function.argmax(state)

    def test_extract_policy(self, q_function):
        for batch_size in [16, 1]:
            if q_function.discrete_action:
                policy = q_function.extract_policy(temperature=10.)
                state = random_state(q_function, batch_size)
                action = random_action(q_function, batch_size)
                sample_action = policy(state).sample()
                assert sample_action.shape == action.shape
                assert sample_action.dtype is torch.long
            else:
                with pytest.raises(NotImplementedError):
                    q_function.extract_policy(temperature=10.)


def random_state(value_function, batch_size):
    if value_function.discrete_state:
        if batch_size > 1:
            return torch.randint(value_function.num_states, (batch_size, 1))
        else:
            return torch.randint(value_function.num_states, (1,))
    else:
        if batch_size > 1:
            return torch.randn(batch_size, value_function.dim_state)
        else:
            return torch.randn(value_function.dim_state)


def random_action(q_function, batch_size):
    if q_function.discrete_action:
        if batch_size > 1:
            return torch.randint(q_function.num_actions, (batch_size,))
        else:
            return torch.randint(q_function.num_actions, ())
    else:
        if batch_size > 1:
            return torch.randn(batch_size, q_function.dim_action)
        else:
            return torch.randn(q_function.dim_action)
