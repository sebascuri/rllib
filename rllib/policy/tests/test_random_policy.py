from rllib.policy import RandomPolicy
from rllib.util.neural_networks import random_tensor
from rllib.util.utilities import tensor_to_distribution
from torch.distributions import MultivariateNormal, Categorical
import pytest


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


class TestRandomPolicy(object):
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

        self.policy = RandomPolicy(self.dim_state, self.dim_action,
                                   self.num_states, self.num_actions)

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
