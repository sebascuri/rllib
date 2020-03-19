import pytest
import torch.testing

from rllib.algorithms.tabular_planning import iterative_policy_evaluation, \
    policy_iteration, \
    value_iteration, linear_system_policy_evaluation
from rllib.environment import EasyGridWorld
from rllib.environment.gym_environment import GymEnvironment
from rllib.policy import RandomPolicy


def test_iterative_policy_evaluation():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3

    policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                          num_actions=environment.num_actions)
    value_function = iterative_policy_evaluation(policy, environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([3.3, 8.8, 4.4, 5.3, 1.5,
                                                1.5, 3.0, 2.3, 1.9, 0.5,
                                                0.1, 0.7, 0.7, 0.4, -0.4,
                                                -1.0, -0.4, -0.4, -0.6, -1.2,
                                                -1.9, -1.3, -1.2, -1.4, -2.0]),
                                  atol=0.05, rtol=EPS)


def test_linear_system_policy_evaluation():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3

    policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                          num_actions=environment.num_actions)
    value_function = linear_system_policy_evaluation(policy, environment, GAMMA)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([3.3, 8.8, 4.4, 5.3, 1.5,
                                                1.5, 3.0, 2.3, 1.9, 0.5,
                                                0.1, 0.7, 0.7, 0.4, -0.4,
                                                -1.0, -0.4, -0.4, -0.6, -1.2,
                                                -1.9, -1.3, -1.2, -1.4, -2.0]),
                                  atol=0.05, rtol=EPS)


def test_policy_iteration():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = policy_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([22.0, 24.4, 22.0, 19.4, 17.5,
                                                19.8, 22.0, 19.8, 17.8, 16.0,
                                                17.8, 19.8, 17.8, 16.0, 14.4,
                                                16.0, 17.8, 16.0, 14.4, 13.0,
                                                14.4, 16.0, 14.4, 13.0, 11.7]),
                                  atol=0.05, rtol=EPS)
    op = [2, 3, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 1, 3]
    opt = policy.table.argmax(dim=0)

    for state in range(environment.num_states):
        environment.state = state
        ns, r, done, info = environment.step(op[state])
        nv = r + GAMMA * value_function.table[:, ns]

        environment.state = state
        ns, r, done, info = environment.step(opt[state])
        nv2 = r + GAMMA * value_function.table[:, ns]

        torch.testing.assert_allclose(nv, nv2)

    environment = EasyGridWorld(terminal_states=[22])
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = policy_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([22.0, 24.4, 22.0, 19.4, 17.5,
                                                19.8, 22.0, 19.8, 17.8, 16.0,
                                                17.8, 19.8, 17.8, 16.0, 14.4,
                                                16.0, 17.8, 16.0, 14.4, 13.0,
                                                14.4, 16.0, 0.0, 13.0, 11.7]),
                                  atol=0.05, rtol=EPS)

    op = [2, 3, 3, 3, 3, 2, 1, 1, 3, 3, 2, 1, 1, 3, 3, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1]
    opt = policy.table.argmax(dim=0)
    for state in range(environment.num_states):  # Test optimality by value.
        environment.state = state
        ns, r, done, info = environment.step(op[state])
        nv = r + GAMMA * value_function.table[:, ns]

        environment.state = state
        ns, r, done, info = environment.step(opt[state])
        nv2 = r + GAMMA * value_function.table[:, ns]

        torch.testing.assert_allclose(nv, nv2)


def test_value_iteration():
    environment = EasyGridWorld()
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = value_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([22.0, 24.4, 22.0, 19.4, 17.5,
                                                19.8, 22.0, 19.8, 17.8, 16.0,
                                                17.8, 19.8, 17.8, 16.0, 14.4,
                                                16.0, 17.8, 16.0, 14.4, 13.0,
                                                14.4, 16.0, 14.4, 13.0, 11.7]),
                                  atol=0.05, rtol=EPS)
    torch.testing.assert_allclose(policy.table.argmax(dim=0),
                                  torch.tensor([2, 3, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 3, 3]))

    environment = EasyGridWorld(terminal_states=[22])
    GAMMA = 0.9
    EPS = 1e-3
    policy, value_function = value_iteration(environment, GAMMA, eps=EPS)

    torch.testing.assert_allclose(value_function.table,
                                  torch.tensor([22.0, 24.4, 22.0, 19.4, 17.5,
                                                19.8, 22.0, 19.8, 17.8, 16.0,
                                                17.8, 19.8, 17.8, 16.0, 14.4,
                                                16.0, 17.8, 16.0, 14.4, 13.0,
                                                14.4, 16.0, 0.0, 13.0, 11.7]),
                                  atol=0.05, rtol=EPS)
    torch.testing.assert_allclose(policy.table.argmax(dim=0),
                                  torch.tensor([2, 3, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 3, 3,
                                                2, 1, 3, 1, 3]))


def test_not_implemented():
    environment = GymEnvironment('CartPole-v0')
    with pytest.raises(NotImplementedError):
        iterative_policy_evaluation(0, environment, 0.9)
    with pytest.raises(NotImplementedError):
        value_iteration(environment, 0.9)
    with pytest.raises(NotImplementedError):
        policy_iteration(environment, 0.9)
