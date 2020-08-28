import pytest

from rllib.environment import GymEnvironment
from rllib.value_function import (
    IntegrateQValueFunction,
    NNEnsembleQFunction,
    NNEnsembleValueFunction,
    NNQFunction,
    NNValueFunction,
    TabularQFunction,
    TabularValueFunction,
)


@pytest.fixture(
    params=["CartPole-v0", "Pendulum-v0", "MountainCarContinuous-v0", "Taxi-v3"]
)
def environment(request):
    return GymEnvironment(request.param)


@pytest.fixture(params=["Taxi-v3"])
def discrete_environment(request):
    return GymEnvironment(request.param)


@pytest.fixture(
    params=[NNEnsembleValueFunction, NNValueFunction, IntegrateQValueFunction]
)
def value_function(request):
    return request.param


@pytest.fixture(params=[NNEnsembleQFunction, NNQFunction])
def q_function(request):
    return request.param


def test_value_function_creation(value_function, environment):
    value_function = value_function.default(environment)
    assert value_function.num_states == environment.num_states
    assert value_function.dim_state == environment.dim_state


def test_q_function_creation(q_function, environment):
    q_function = q_function.default(environment)
    assert q_function.num_states == environment.num_states
    assert q_function.dim_state == environment.dim_state
    assert q_function.num_actions == environment.num_actions
    assert q_function.dim_action == environment.dim_action


def test_tabular_creation(discrete_environment):
    test_value_function_creation(TabularValueFunction, environment=discrete_environment)
    test_q_function_creation(TabularQFunction, environment=discrete_environment)
