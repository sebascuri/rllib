import pytest
import torch
import torch.testing

from rllib.policy import NNPolicy
from rllib.util.neural_networks import random_tensor
from rllib.value_function import (
    IntegrateQValueFunction,
    NNEnsembleQFunction,
    NNQFunction,
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


@pytest.fixture(params=[None, 1, 5])
def num_heads(request):
    return request.param


@pytest.fixture(params=[None, 1, 16])
def batch_size(request):
    return request.param


class TestIntegrateValueFunction(object):
    def init(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        num_heads,
        num_samples=1,
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

        if num_heads is None:
            self.q_function = NNQFunction(
                dim_state=self.dim_state,
                dim_action=self.dim_action,
                num_states=self.num_states,
                num_actions=self.num_actions,
                layers=layers,
                biased_head=biased_head,
            )
        else:
            self.q_function = NNEnsembleQFunction(
                dim_state=self.dim_state,
                dim_action=self.dim_action,
                num_states=self.num_states,
                num_actions=self.num_actions,
                num_heads=num_heads,
                layers=layers,
                biased_head=biased_head,
            )

        self.policy = NNPolicy(
            dim_state=self.dim_state,
            dim_action=self.dim_action,
            num_states=self.num_states,
            num_actions=self.num_actions,
            layers=layers,
            biased_head=biased_head,
        )

        self.value_function = IntegrateQValueFunction(
            q_function=self.q_function, policy=self.policy, num_samples=num_samples
        )

    def test_property_values(
        self, discrete_state, discrete_action, dim_state, dim_action, num_heads
    ):
        if not (discrete_state and not discrete_action):
            self.init(discrete_state, discrete_action, dim_state, dim_action, num_heads)
            assert (
                self.num_states if self.num_states is not None else -1
            ) == self.value_function.num_states

            assert discrete_state == self.value_function.discrete_state
            assert self.dim_state == self.value_function.dim_state

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
            value = self.value_function(state)

            if num_heads:
                assert value.shape == torch.Size(
                    [batch_size, num_heads] if batch_size else [num_heads]
                )
            else:
                assert value.shape == torch.Size([batch_size] if batch_size else [])
            assert value.dtype is torch.get_default_dtype()
