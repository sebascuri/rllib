import pytest
import torch
import torch.nn as nn
import torch.testing

from rllib.util.neural_networks import random_tensor
from rllib.value_function import (
    NNEnsembleQFunction,
    NNEnsembleValueFunction,
    NNQFunction,
    NNValueFunction,
)


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


@pytest.fixture(params=[1, 5])
def num_heads(request):
    return request.param


@pytest.fixture(params=[None, 1, 16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[True, False])
def biased_head(request):
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


class TestNNEnsembleValueFunction(object):
    def init(self, discrete_state, dim_state, num_heads, layers=None, biased_head=True):
        self.num_states, self.dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )
        layers = layers if layers is not None else [32, 32]

        self.value_function = NNEnsembleValueFunction(
            dim_state=self.dim_state,
            num_states=self.num_states,
            num_heads=num_heads,
            layers=layers,
            biased_head=biased_head,
        )

    def test_compile(self, discrete_state, dim_state, num_heads):
        self.init(discrete_state, dim_state, num_heads)
        torch.jit.script(self.value_function)

    def test_property_values(self, discrete_state, dim_state, num_heads):
        self.init(discrete_state, dim_state, num_heads)
        assert (
            self.num_states if self.num_states is not None else -1
        ) == self.value_function.num_states
        assert discrete_state == self.value_function.discrete_state
        assert self.dim_state == self.value_function.dim_state

    def test_forward(self, discrete_state, dim_state, num_heads, batch_size):
        self.init(discrete_state, dim_state, num_heads)
        state = random_tensor(discrete_state, dim_state, batch_size)
        value = self.value_function(state)

        assert value.shape == torch.Size(
            [batch_size, num_heads] if batch_size else [num_heads]
        )
        assert value.dtype is torch.get_default_dtype()

    def test_embeddings(
        self, discrete_state, dim_state, num_heads, batch_size, biased_head
    ):
        layers = [64, 64]
        self.init(
            discrete_state, dim_state, num_heads, layers=layers, biased_head=biased_head
        )
        dim = layers[-1] + 1 if biased_head else layers[-1]
        state = random_tensor(discrete_state, dim_state, batch_size)
        embeddings = self.value_function.embeddings(state)

        assert embeddings.shape == torch.Size(
            [batch_size, dim, num_heads] if batch_size else [dim, num_heads]
        )
        assert embeddings.dtype is torch.get_default_dtype()

    def test_input_transform(self, num_heads, batch_size):
        value_function = NNEnsembleValueFunction(
            dim_state=(2,),
            num_heads=num_heads,
            layers=[64, 64],
            non_linearity="Tanh",
            input_transform=StateTransform(),
        )
        value = value_function(random_tensor(False, 2, batch_size))

        assert value.shape == torch.Size(
            [batch_size, num_heads] if batch_size else [num_heads]
        )
        assert value.dtype is torch.get_default_dtype()

    def test_from_value_function(self, discrete_state, dim_state, num_heads):
        num_states, dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )

        value_function = NNValueFunction(dim_state=dim_state, num_states=num_states)

        other = NNEnsembleValueFunction.from_value_function(value_function, num_heads)

        assert value_function is not other
        assert other.num_heads == num_heads


class TestNNEnsembleQFunction(object):
    def init(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        num_heads,
        layers=None,
        biased_head=True,
    ):
        self.num_states, self.dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )
        self.num_actions, self.dim_action = (
            (dim_action, ()) if discrete_action else (-1, (dim_action,))
        )

        layers = layers if layers is not None else [32, 32]

        self.q_function = NNEnsembleQFunction(
            dim_state=self.dim_state,
            dim_action=self.dim_action,
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_heads=num_heads,
            layers=layers,
            biased_head=biased_head,
        )

    def test_compile(
        self, discrete_state, discrete_action, dim_state, dim_action, num_heads
    ):
        if discrete_state and not discrete_action:
            return
        self.init(discrete_state, discrete_action, dim_state, dim_action, num_heads)
        torch.jit.script(self.q_function)

    def test_init(
        self, discrete_state, discrete_action, dim_state, dim_action, num_heads
    ):
        if discrete_state and not discrete_action:
            with pytest.raises(NotImplementedError):
                self.init(
                    discrete_state, discrete_action, dim_state, dim_action, num_heads
                )

    def test_property_values(
        self, discrete_state, discrete_action, dim_state, dim_action, num_heads
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action, num_heads)
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

            assert num_heads == self.q_function.num_heads

    def test_forward(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        num_heads,
        batch_size,
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action, num_heads)
            state = random_tensor(discrete_state, dim_state, batch_size)
            action = random_tensor(discrete_action, dim_action, batch_size)
            value = self.q_function(state, action)

            assert value.shape == torch.Size(
                [batch_size, num_heads] if batch_size else [num_heads]
            )
            assert value.dtype is torch.get_default_dtype()

    def test_input_transform(self, num_heads, batch_size):
        q_function = NNEnsembleQFunction(
            dim_state=(2,),
            dim_action=(1,),
            num_heads=num_heads,
            layers=[64, 64],
            non_linearity="Tanh",
            input_transform=StateTransform(),
        )
        value = q_function(
            random_tensor(False, 2, batch_size), random_tensor(False, 1, batch_size)
        )
        assert value.shape == torch.Size(
            [batch_size, num_heads] if batch_size else [num_heads]
        )
        assert value.dtype is torch.get_default_dtype()

    def test_partial_q_function(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        num_heads,
        batch_size,
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action, num_heads)
            state = random_tensor(discrete_state, dim_state, batch_size)

            if not discrete_action:
                with pytest.raises(NotImplementedError):
                    self.q_function(state)
            else:
                action_value = self.q_function(state)

                assert action_value.shape == torch.Size(
                    [batch_size, self.num_actions, num_heads]
                    if batch_size
                    else [self.num_actions, num_heads]
                )

    def test_from_q_function(
        self, discrete_state, discrete_action, dim_state, dim_action, num_heads
    ):
        num_states, dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )
        num_actions, dim_action = (
            (dim_action, ()) if discrete_action else (-1, (dim_action,))
        )

        if not (discrete_state and not discrete_action):
            q_function = NNQFunction(
                dim_state=dim_state,
                num_states=num_states,
                dim_action=dim_action,
                num_actions=num_actions,
            )

            other = NNEnsembleQFunction.from_q_function(q_function, num_heads)

            assert q_function is not other
            assert other.num_heads == num_heads
