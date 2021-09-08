import pytest
import torch.testing

from rllib.algorithms.tabular_planning import (
    iterative_policy_evaluation,
    linear_system_policy_evaluation,
    policy_iteration,
    value_iteration,
)
from rllib.environment.gym_environment import GymEnvironment
from rllib.environment.mdps import EasyGridWorld
from rllib.policy import RandomPolicy

RANDOM_VALUE = (
    [3.3, 8.8, 4.4, 5.3, 1.5]
    + [1.5, 3.0, 2.3, 1.9, 0.5]
    + [0.1, 0.7, 0.7, 0.4, -0.4]
    + [-1.0, -0.4, -0.4, -0.6, -1.2]
    + [-1.9, -1.3, -1.2, -1.4, -2.0]
)

OPTIMAL_VALUE = (
    [22.0, 24.4, 22.0, 19.4, 17.5]
    + [19.8, 22.0, 19.8, 17.8, 16.0]
    + [17.8, 19.8, 17.8, 16.0, 14.4]
    + [16.0, 17.8, 16.0, 14.4, 13.0]
    + [14.4, 16.0, 14.4, 13.0, 11.7]
)

OPTIMAL_VALUE_WITH_TERMINAL = (
    [22.0, 24.4, 22.0, 19.4, 17.5]
    + [19.8, 22.0, 19.8, 17.8, 16.0]
    + [17.8, 19.8, 17.8, 16.0, 14.4]
    + [16.0, 17.8, 16.0, 14.4, 13.0]
    + [14.4, 16.0, 0.0, 13.0, 11.7]
)
OPTIMAL_POLICY = (
    [2, 3, 3, 3, 3]
    + [2, 1, 3, 3, 3]
    + [2, 1, 3, 3, 3]
    + [2, 1, 3, 3, 3]
    + [2, 1, 3, 1, 3]
)

OPTIMAL_POLICY_WITH_TERMINAL = (
    [2, 3, 3, 3, 3]
    + [2, 1, 1, 3, 3]
    + [2, 1, 1, 3, 3]
    + [2, 1, 1, 1, 1]
    + [2, 1, 3, 1, 1]
)


def test_iterative_policy_evaluation():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3

    policy = RandomPolicy(
        dim_state=(),
        dim_action=(),
        num_states=environment.num_states,
        num_actions=environment.num_actions,
    )
    value_function = iterative_policy_evaluation(policy, environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(
        value_function.table, torch.tensor([RANDOM_VALUE]), atol=0.05, rtol=EPS
    )


def test_linear_system_policy_evaluation():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3

    policy = RandomPolicy(
        dim_state=(),
        dim_action=(),
        num_states=environment.num_states,
        num_actions=environment.num_actions,
    )
    value_function = linear_system_policy_evaluation(policy, environment, GAMMA)

    torch.testing.assert_allclose(
        value_function.table, torch.tensor([RANDOM_VALUE]), atol=0.05, rtol=EPS
    )


def test_policy_iteration():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = policy_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(
        value_function.table, torch.tensor([OPTIMAL_VALUE]), atol=0.05, rtol=EPS
    )
    pred_p = policy.table.argmax(dim=0)
    assert_policy_equality(environment, GAMMA, value_function, OPTIMAL_POLICY, pred_p)

    environment = EasyGridWorld(terminal_states=[22])
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = policy_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(
        value_function.table,
        torch.tensor([OPTIMAL_VALUE_WITH_TERMINAL]),
        atol=0.05,
        rtol=EPS,
    )

    pred_p = policy.table.argmax(dim=0)
    assert_policy_equality(
        environment, GAMMA, value_function, OPTIMAL_POLICY_WITH_TERMINAL, pred_p
    )


def test_value_iteration():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = value_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(
        value_function.table, torch.tensor([OPTIMAL_VALUE]), atol=0.05, rtol=EPS
    )
    pred_p = policy.table.argmax(dim=0)
    assert_policy_equality(environment, GAMMA, value_function, OPTIMAL_POLICY, pred_p)

    environment = EasyGridWorld(terminal_states=[22])
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = value_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(
        value_function.table,
        torch.tensor([OPTIMAL_VALUE_WITH_TERMINAL]),
        atol=0.05,
        rtol=EPS,
    )

    pred_p = policy.table.argmax(dim=0)
    assert_policy_equality(
        environment, GAMMA, value_function, OPTIMAL_POLICY_WITH_TERMINAL, pred_p
    )


def assert_policy_equality(environment, gamma, value_function, true_opt_p, pred_opt_p):
    """Assert equality by checking Bellman operator equality."""
    for state in range(environment.num_states):
        environment.state = state
        next_state, reward, done, info = environment.step(pred_opt_p[state])
        pred_value = reward + gamma * value_function(torch.tensor(next_state))

        environment.state = state
        next_state, reward, done, info = environment.step(true_opt_p[state])
        true_value = reward + gamma * value_function(torch.tensor(next_state))
        torch.testing.assert_allclose(pred_value, true_value)


def test_not_implemented():
    environment = GymEnvironment("CartPole-v0")
    with pytest.raises(AttributeError):
        iterative_policy_evaluation(0, environment, 0.9)  # type: ignore
    with pytest.raises(AttributeError):
        value_iteration(environment, 0.9)  # type: ignore
    with pytest.raises(AttributeError):
        policy_iteration(environment, 0.9)  # type: ignore
