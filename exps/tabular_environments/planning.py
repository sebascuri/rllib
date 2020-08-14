"""Tabular planning experiments."""
from rllib.algorithms.tabular_planning import (
    iterative_policy_evaluation,
    linear_system_policy_evaluation,
    policy_iteration,
    value_iteration,
)
from rllib.environment.mdps import EasyGridWorld
from rllib.policy import AbstractPolicy, RandomPolicy

environment = EasyGridWorld()
GAMMA = 0.9
EPS = 1e-6

policy = RandomPolicy(
    dim_state=(),
    dim_action=(),
    num_states=environment.num_states,
    num_actions=environment.num_actions,
)  # type: AbstractPolicy

print("Iterative Policy Evaluation:")
value_function = iterative_policy_evaluation(policy, environment, GAMMA, eps=EPS)
print(value_function.table)
print()

print("Linear System Policy Evaluation:")
value_function = linear_system_policy_evaluation(policy, environment, GAMMA)
print(value_function.table)
print()

print("Policy Iteration:")
policy, value_function = policy_iteration(environment, GAMMA, eps=EPS)
print(policy.table.argmax(dim=0))
print(value_function.table)
print()

print("Value Iteration")
policy, value_function = value_iteration(environment, GAMMA, eps=EPS)
print(policy.table.argmax(dim=0))
print(value_function.table)
print()

print("Iterative Policy Evaluation from Value Iteration:")
value_function = iterative_policy_evaluation(
    policy, environment, GAMMA, eps=EPS, value_function=value_function
)
print(value_function.table)
print()


print("Linear System Policy Evaluation from Value Iteration:")
value_function = linear_system_policy_evaluation(policy, environment, GAMMA)
print(value_function.table)
print()
