import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

from rllib.policy import FelixPolicy, NNPolicy
from rllib.util.distributions import Delta
from rllib.util.neural_networks import HomoGaussianNN, count_vars, random_tensor
from rllib.util.utilities import tensor_to_distribution


@pytest.fixture(params=[True, False])
def discrete_state(request):
    return request.param


@pytest.fixture(params=[True, False])
def discrete_action(request):
    return request.param


@pytest.fixture(params=[1, 4])
def dim_state(request):
    return request.param


@pytest.fixture(params=[1, 4])
def dim_action(request):
    return request.param


@pytest.fixture(params=[None, 1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[True, False])
def deterministic(request):
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


class TestMLPPolicy(object):
    def init(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        deterministic=False,
    ):

        self.num_states, self.dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )

        self.num_actions, self.dim_action = (
            (dim_action, ()) if discrete_action else (-1, (dim_action,))
        )

        self.policy = NNPolicy(
            dim_state=self.dim_state,
            dim_action=self.dim_action,
            num_states=self.num_states,
            num_actions=self.num_actions,
            layers=[32, 32],
            deterministic=deterministic,
        )

    def test_property_values(
        self, discrete_state, discrete_action, dim_state, dim_action
    ):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        assert (
            self.num_states if self.num_states is not None else -1
        ) == self.policy.num_states
        assert (
            self.num_actions if self.num_actions is not None else -1
        ) == self.policy.num_actions
        assert discrete_state == self.policy.discrete_state
        assert discrete_action == self.policy.discrete_action

    def test_random_action(
        self, discrete_state, discrete_action, dim_state, dim_action
    ):
        self.init(discrete_state, discrete_action, dim_state, dim_action)

        distribution = tensor_to_distribution(self.policy.random())
        sample = distribution.sample()

        if distribution.has_enumerate_support:  # Discrete
            assert distribution.logits.shape == (self.num_actions,)
            assert sample.shape == ()
        else:  # Continuous
            assert distribution.mean.shape == self.dim_action
            assert sample.shape == (dim_action,)

    def test_forward(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        batch_size,
        deterministic,
    ):
        self.init(discrete_state, discrete_action, dim_state, dim_action, deterministic)
        state = random_tensor(discrete_state, dim_state, batch_size)
        distribution = tensor_to_distribution(self.policy(state))
        sample = distribution.sample()

        if distribution.has_enumerate_support:  # Discrete
            assert isinstance(distribution, Categorical)
            if batch_size:
                assert distribution.logits.shape == (batch_size, self.num_actions)
                assert sample.shape == (batch_size,)
            else:
                assert distribution.logits.shape == (self.num_actions,)
                assert sample.shape == ()
        else:  # Continuous
            if deterministic:
                assert isinstance(distribution, Delta)
            else:
                assert isinstance(distribution, MultivariateNormal)

            if batch_size:
                assert distribution.mean.shape == (batch_size,) + self.dim_action
                if not deterministic:
                    assert distribution.covariance_matrix.shape == (
                        batch_size,
                        self.dim_action[0],
                        self.dim_action[0],
                    )
                assert sample.shape == (batch_size, dim_action)
            else:
                assert distribution.mean.shape == self.dim_action
                if not deterministic:
                    assert distribution.covariance_matrix.shape == (
                        self.dim_action[0],
                        self.dim_action[0],
                    )
                assert sample.shape == (dim_action,)

    def test_embeddings(
        self, discrete_state, discrete_action, dim_state, dim_action, batch_size
    ):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        state = random_tensor(discrete_state, dim_state, batch_size)
        embeddings = self.policy.embeddings(state)

        assert embeddings.shape == torch.Size([batch_size, 33] if batch_size else [33])
        assert embeddings.dtype is torch.get_default_dtype()

    def test_input_transform(self, batch_size):
        policy = NNPolicy(
            dim_state=(2,),
            dim_action=(4,),
            layers=[64, 64],
            input_transform=StateTransform(),
        )
        out = tensor_to_distribution(policy(random_tensor(False, 2, batch_size)))
        action = out.sample()
        assert action.shape == torch.Size([batch_size, 4] if batch_size else [4])
        assert action.dtype is torch.get_default_dtype()

    def test_goal(self, batch_size):
        goal = random_tensor(False, 3, None)
        policy = NNPolicy(dim_state=(4,), dim_action=(2,), layers=[32, 32], goal=goal)
        state = random_tensor(False, 4, batch_size)
        pi = tensor_to_distribution(policy(state))
        action = pi.sample()
        assert action.shape == torch.Size([batch_size, 2] if batch_size else [2])
        assert action.dtype is torch.get_default_dtype()

        other_goal = random_tensor(False, 3, None)
        policy.set_goal(other_goal)
        other_pi = tensor_to_distribution(policy(state))

        assert not torch.any(other_pi.mean == pi.mean)

    def test_from_other(self, discrete_state, discrete_action, dim_state, dim_action):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        _test_from_other(self.policy, NNPolicy)
        _test_from_other_with_copy(self.policy, NNPolicy)

    def test_from_nn(self, discrete_state, dim_state, dim_action, batch_size):
        self.init(discrete_state, False, dim_state, dim_action)
        policy = NNPolicy.from_nn(
            HomoGaussianNN(
                self.policy.nn.kwargs["in_dim"],
                self.policy.nn.kwargs["out_dim"],
                layers=[20, 20],
                biased_head=False,
            ),
            self.dim_state,
            self.dim_action,
            num_states=self.num_states,
            num_actions=self.num_actions,
        )

        state = random_tensor(discrete_state, dim_state, batch_size)
        action = tensor_to_distribution(policy(state)).sample()
        embeddings = policy.embeddings(state)

        assert action.shape == torch.Size(
            [batch_size, dim_action] if batch_size else [dim_action]
        )
        assert embeddings.shape == torch.Size([batch_size, 20] if batch_size else [20])
        assert action.dtype is torch.get_default_dtype()
        assert embeddings.dtype is torch.get_default_dtype()


class TestFelixNet(object):
    def init(self, dim_state, dim_action):
        self.policy = FelixPolicy(dim_state=(dim_state,), dim_action=(dim_action,))

    def test_num_states_actions(self, dim_state, dim_action):
        self.init(dim_state, dim_action)
        assert self.policy.num_states == -1
        assert self.policy.num_actions == -1
        assert not self.policy.discrete_state
        assert not self.policy.discrete_action
        assert self.policy.dim_state == (dim_state,)
        assert self.policy.dim_action == (dim_action,)

    def test_random_action(self, dim_state, dim_action):
        self.init(dim_state, dim_action)

        distribution = tensor_to_distribution(self.policy.random())
        sample = distribution.sample()

        assert distribution.mean.shape == (dim_action,)
        assert sample.shape == (dim_action,)

    def test_call(self, dim_state, dim_action, batch_size):
        self.init(dim_state, dim_action)
        state = random_tensor(False, dim_state, batch_size)

        distribution = tensor_to_distribution(self.policy(state))
        sample = distribution.sample()

        assert isinstance(distribution, MultivariateNormal)
        if batch_size:
            assert distribution.mean.shape == (batch_size, dim_action)
            assert distribution.covariance_matrix.shape == (
                batch_size,
                dim_action,
                dim_action,
            )
            assert sample.shape == (batch_size, dim_action)
        else:
            assert distribution.mean.shape == (dim_action,)
            assert distribution.covariance_matrix.shape == (dim_action, dim_action)
            assert sample.shape == (dim_action,)
