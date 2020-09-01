import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.testing

from rllib.util.neural_networks import DeterministicNN, count_vars, random_tensor
from rllib.value_function import DuelingQFunction, NNQFunction, NNValueFunction


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


class StateTransform(nn.Module):
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1
        )
        return states_


def _test_from_other(object_, class_):
    other = class_.from_other(object_, copy=False)

    assert isinstance(other, class_)
    assert other is not object_

    other_state_dict = other.state_dict()
    for name, param in object_.named_parameters():
        if not torch.allclose(param, torch.zeros_like(param)):
            assert not torch.allclose(param, other_state_dict[name])
    assert count_vars(other) == count_vars(object_)


def _test_from_other_with_copy(object_, class_):
    other = class_.from_other(object_, copy=True)

    assert isinstance(other, class_)
    assert other is not object_
    other_state_dict = other.state_dict()

    for name, param in object_.named_parameters():
        assert torch.allclose(param, other_state_dict[name])
    assert count_vars(other) == count_vars(object_)


class TestNNValueFunction(object):
    def init(self, discrete_state, dim_state):
        if discrete_state:
            self.num_states = dim_state
            self.dim_state = ()
        else:
            self.num_states = -1
            self.dim_state = (dim_state,)

        layers = [32, 32]
        self.value_function = NNValueFunction(
            dim_state=self.dim_state, num_states=self.num_states, layers=layers
        )

    def test_compile(self, discrete_state, dim_state):
        self.init(discrete_state, dim_state)
        torch.jit.script(self.value_function)

    def test_property_values(self, discrete_state, dim_state):
        self.init(discrete_state, dim_state)
        assert (
            self.num_states if self.num_states is not None else -1
        ) == self.value_function.num_states
        assert discrete_state == self.value_function.discrete_state
        assert self.dim_state == self.value_function.dim_state

    def test_forward(self, discrete_state, dim_state, batch_size):
        self.init(discrete_state, dim_state)
        state = random_tensor(discrete_state, dim_state, batch_size)
        value = self.value_function(state)

        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()

    def test_embeddings(self, discrete_state, dim_state, batch_size):
        self.init(discrete_state, dim_state)
        state = random_tensor(discrete_state, dim_state, batch_size)
        embeddings = self.value_function.embeddings(state)

        assert embeddings.shape == torch.Size([batch_size, 33] if batch_size else [33])
        assert embeddings.dtype is torch.get_default_dtype()

    def test_input_transform(self, batch_size):
        value_function = NNValueFunction(
            dim_state=(2,),
            num_states=-1,
            layers=[64, 64],
            non_linearity="Tanh",
            input_transform=StateTransform(),
        )
        value = value_function(random_tensor(False, 2, batch_size))
        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()

    def test_from_other(self, discrete_state, dim_state):
        self.init(discrete_state, dim_state)
        _test_from_other(self.value_function, NNValueFunction)
        _test_from_other_with_copy(self.value_function, NNValueFunction)

    def test_from_nn(self, discrete_state, dim_state, batch_size):
        self.init(discrete_state, dim_state)
        value_function = torch.jit.script(
            NNValueFunction.from_nn(
                DeterministicNN(
                    self.value_function.nn.kwargs["in_dim"],
                    self.value_function.nn.kwargs["out_dim"],
                    layers=[20, 20],
                    biased_head=False,
                ),
                self.dim_state,
                num_states=self.num_states,
            )
        )

        state = random_tensor(discrete_state, dim_state, batch_size)
        value = value_function(state)
        embeddings = value_function.embeddings(state)

        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert embeddings.shape == torch.Size([batch_size, 20] if batch_size else [20])
        assert value.dtype is torch.get_default_dtype()
        assert embeddings.dtype is torch.get_default_dtype()


class TestNNQFunction(object):
    base_class = NNQFunction

    def init(self, discrete_state, discrete_action, dim_state, dim_action):
        if discrete_state:
            self.num_states = dim_state
            self.dim_state = ()
        else:
            self.num_states = -1
            self.dim_state = (dim_state,)

        if discrete_action:
            self.num_actions = dim_action
            self.dim_action = ()
        else:
            self.num_actions = -1
            self.dim_action = (dim_action,)

        layers = [32, 32]
        self.q_function = self.base_class(
            dim_state=self.dim_state,
            dim_action=self.dim_action,
            num_states=self.num_states,
            num_actions=self.num_actions,
            layers=layers,
        )

    def test_compile(self, discrete_state, discrete_action, dim_state, dim_action):
        if discrete_state and not discrete_action:
            return
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        torch.jit.script(self.q_function)

    def test_init(self, discrete_state, discrete_action, dim_state, dim_action):
        if discrete_state and not discrete_action:
            with pytest.raises(NotImplementedError):
                self.init(discrete_state, discrete_action, dim_state, dim_action)

    def test_input_transform(self, batch_size):
        q_function = NNQFunction(
            dim_state=(2,),
            dim_action=(1,),
            layers=[64, 64],
            non_linearity="Tanh",
            input_transform=StateTransform(),
        )
        value = q_function(
            random_tensor(False, 2, batch_size), random_tensor(False, 1, batch_size)
        )
        assert value.shape == torch.Size([batch_size] if batch_size else [])
        assert value.dtype is torch.get_default_dtype()

    def test_property_values(
        self, discrete_state, discrete_action, dim_state, dim_action
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            assert (
                self.num_states if self.num_states is not None else -1
            ) == self.q_function.num_states
            assert (
                self.num_actions if self.num_actions is not None else -1
            ) == self.q_function.num_actions

            assert discrete_state == self.q_function.discrete_state
            assert discrete_action == self.q_function.discrete_action
            assert self.dim_state == self.q_function.dim_state
            assert self.dim_action == self.q_function.dim_action

    def test_forward(
        self, discrete_state, discrete_action, dim_state, dim_action, batch_size
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            state = random_tensor(discrete_state, dim_state, batch_size)
            action = random_tensor(discrete_action, dim_action, batch_size)
            value = self.q_function(state, action)
            assert value.shape == torch.Size([batch_size] if batch_size else [])
            assert value.dtype is torch.get_default_dtype()

    def test_partial_q_function(
        self, discrete_state, discrete_action, dim_state, dim_action, batch_size
    ):
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

    def test_from_other(self, discrete_state, discrete_action, dim_state, dim_action):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            _test_from_other(self.q_function, NNQFunction)
            _test_from_other_with_copy(self.q_function, NNQFunction)

    def test_from_nn(
        self, discrete_state, discrete_action, dim_state, dim_action, batch_size
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action)
            q_function = NNQFunction.from_nn(
                nn.Linear(
                    self.q_function.nn.kwargs["in_dim"][0],
                    self.q_function.nn.kwargs["out_dim"][0],
                ),
                self.dim_state,
                self.dim_action,
                num_states=self.num_states,
                num_actions=self.num_actions,
            )

            state = random_tensor(discrete_state, dim_state, batch_size)
            action = random_tensor(discrete_action, dim_action, batch_size)
            value = q_function(state, action)
            assert value.shape == torch.Size([batch_size] if batch_size else [])
            assert value.dtype is torch.get_default_dtype()


class TestDuelingQFunction(TestNNQFunction):
    base_class = DuelingQFunction  # type: ignore

    def init(self, discrete_state, discrete_action, dim_state, dim_action):
        super().init(discrete_state, discrete_action, dim_state, dim_action)
        self.q_function.average_or_mean = np.random.choice(["average", "mean"])

    @pytest.fixture(params=[True], scope="class")
    def discrete_action(self, request):
        return request.param

    def test_compile(self, discrete_state, discrete_action, dim_state, dim_action):
        pass
