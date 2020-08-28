import pytest

from rllib.environment import GymEnvironment
from rllib.policy import (  # MPCPolicy,
    DerivedPolicy,
    EpsGreedy,
    FelixPolicy,
    MellowMax,
    NNPolicy,
    RandomPolicy,
    SoftMax,
    TabularPolicy,
)


@pytest.fixture(
    params=["CartPole-v0", "Pendulum-v0", "MountainCarContinuous-v0", "Taxi-v3"]
)
def environment(request):
    return GymEnvironment(request.param)


@pytest.fixture(params=[NNPolicy, RandomPolicy])
def policy(request):
    return request.param


@pytest.fixture(params=["CartPole-v0", "Taxi-v3"])
def discrete_action_environment(request):
    return GymEnvironment(request.param)


@pytest.fixture(params=[EpsGreedy, SoftMax, MellowMax])
def discrete_action_policy(request):
    return request.param


@pytest.fixture(params=["Pendulum-v0"])
def continuous_action_environment(request):
    return GymEnvironment(request.param)


@pytest.fixture(params=[FelixPolicy, DerivedPolicy])
def continuous_action_policy(request):
    return request.param


@pytest.fixture(params=["Taxi-v3"])
def discrete_environment(request):
    return GymEnvironment(request.param)


def test_policy_creation(policy, environment):
    policy = policy.default(environment)
    assert policy.num_states == environment.num_states
    assert policy.dim_state == environment.dim_state
    assert policy.num_actions == environment.num_actions
    assert policy.dim_action == environment.dim_action


def test_discrete_action_policy_creation(
    discrete_action_policy, discrete_action_environment
):
    test_policy_creation(discrete_action_policy, discrete_action_environment)


def test_continuous_action_policy_creation(
    continuous_action_policy, continuous_action_environment
):
    test_policy_creation(continuous_action_policy, continuous_action_environment)


def test_tabular_creation(discrete_environment):
    test_policy_creation(TabularPolicy, discrete_environment)
