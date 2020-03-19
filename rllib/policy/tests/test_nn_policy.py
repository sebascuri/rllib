import pytest
import torch
from torch.distributions import MultivariateNormal, Categorical

from rllib.policy import NNPolicy, FelixPolicy, TabularPolicy
from rllib.util.neural_networks import random_tensor
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


class TestMLPPolicy(object):
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

        self.policy = NNPolicy(self.dim_state, self.dim_action,
                               self.num_states, self.num_actions,
                               layers=[32, 32])

    def test_num_states_actions(self, discrete_state, discrete_action, dim_state,
                                dim_action):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        assert self.num_states == self.policy.num_states
        assert self.num_actions == self.policy.num_actions

        if not discrete_state:
            assert self.policy.num_states is None
        if not discrete_action:
            assert self.policy.num_actions is None

        assert discrete_state == self.policy.discrete_state
        assert discrete_action == self.policy.discrete_action

    def test_random_action(self, discrete_state, discrete_action, dim_state,
                           dim_action):
        self.init(discrete_state, discrete_action, dim_state, dim_action)

        distribution = tensor_to_distribution(self.policy.random())
        sample = distribution.sample()

        if distribution.has_enumerate_support:  # Discrete
            assert distribution.logits.shape == (self.num_actions,)
            assert sample.shape == ()
        else:  # Continuous
            assert distribution.mean.shape == (self.dim_action,)
            assert sample.shape == (dim_action,)

    def test_call(self, discrete_state, discrete_action, dim_state, dim_action,
                  batch_size):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        state = random_tensor(discrete_state, dim_state, batch_size)

        distribution = tensor_to_distribution(self.policy(state))
        sample = distribution.sample()

        if distribution.has_enumerate_support:  # Discrete
            assert isinstance(distribution, Categorical)
            if batch_size:
                assert distribution.logits.shape == (batch_size, self.num_actions,)
                assert sample.shape == (batch_size,)
            else:
                assert distribution.logits.shape == (self.num_actions,)
                assert sample.shape == ()
        else:  # Continuous
            assert isinstance(distribution, MultivariateNormal)
            if batch_size:
                assert distribution.mean.shape == (batch_size, self.dim_action,)
                assert distribution.covariance_matrix.shape == (batch_size,
                                                                self.dim_action,
                                                                self.dim_action)
                assert sample.shape == (batch_size, dim_action,)
            else:
                assert distribution.mean.shape == (self.dim_action,)
                assert distribution.covariance_matrix.shape == (self.dim_action,
                                                                self.dim_action)
                assert sample.shape == (dim_action,)

    def test_parameters(self, discrete_state, discrete_action, dim_state, dim_action):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
        old_parameter = self.policy.parameters()
        self.policy.update_parameters(old_parameter)


class TestFelixNet(object):
    def init(self, dim_state, dim_action):
        self.policy = FelixPolicy(dim_state, dim_action)

    def test_discrete(self, dim_state, dim_action):
        with pytest.raises(ValueError):
            FelixPolicy(1, dim_action, num_states=dim_state)
        with pytest.raises(ValueError):
            FelixPolicy(dim_state, 1, num_actions=dim_action)
        with pytest.raises(ValueError):
            FelixPolicy(1, 1, num_states=dim_state, num_actions=dim_action)

    def test_num_states_actions(self, dim_state, dim_action):
        self.init(dim_state, dim_action)
        assert self.policy.num_states is None
        assert self.policy.num_actions is None
        assert not self.policy.discrete_state
        assert not self.policy.discrete_action
        assert self.policy.dim_state == dim_state
        assert self.policy.dim_action == dim_action

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
            assert distribution.mean.shape == (batch_size, dim_action,)
            assert distribution.covariance_matrix.shape == (batch_size,
                                                            dim_action, dim_action)
            assert sample.shape == (batch_size, dim_action,)
        else:
            assert distribution.mean.shape == (dim_action,)
            assert distribution.covariance_matrix.shape == (dim_action, dim_action)
            assert sample.shape == (dim_action,)

    def test_parameters(self, dim_state, dim_action):
        self.init(dim_state, dim_action)
        old_parameter = self.policy.parameters()
        self.policy.update_parameters(old_parameter)


class TestTabularPolicy(object):
    def test_init(self):
        policy = TabularPolicy(num_states=4, num_actions=2)
        torch.testing.assert_allclose(policy.table, 1)

    def test_set_value(self):
        policy = TabularPolicy(num_states=4, num_actions=2)
        policy.set_value(2, torch.tensor(1))
        l1 = torch.log(torch.tensor(1e-12))
        l2 = torch.log(torch.tensor(1. + 1e-12))
        torch.testing.assert_allclose(policy.table,
                                      torch.tensor([[1., 1., l1, 1],
                                                    [1., 1., l2, 1]]))

        policy.set_value(0, torch.tensor([0.3, 0.7]))
        torch.testing.assert_allclose(policy.table,
                                      torch.tensor([[.3, 1., l1, 1],
                                                    [.7, 1., l2, 1]]))
