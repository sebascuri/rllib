import pytest
from torch.distributions import Categorical, MultivariateNormal

from rllib.policy import RandomPolicy
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


class TestRandomPolicy(object):
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

        self.policy = RandomPolicy(
            self.dim_state, self.dim_action, self.num_states, self.num_actions
        )

    def test_num_states_actions(
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

    def test_call(
        self, discrete_state, discrete_action, dim_state, dim_action, batch_size
    ):
        self.init(discrete_state, discrete_action, dim_state, dim_action)
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
            assert isinstance(distribution, MultivariateNormal)
            if batch_size:
                assert distribution.mean.shape == (batch_size,) + self.dim_action
                assert distribution.covariance_matrix.shape == (
                    batch_size,
                    self.dim_action[0],
                    self.dim_action[0],
                )
                assert sample.shape == (batch_size, dim_action)
            else:
                assert distribution.mean.shape == self.dim_action
                assert distribution.covariance_matrix.shape == (
                    self.dim_action[0],
                    self.dim_action[0],
                )
                assert sample.shape == (dim_action,)
