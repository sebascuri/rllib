from rllib.environment.mdp import EasyGridWorld
from rllib.policy import RandomPolicy
from rllib.algorithms import policy_evaluation, policy_iteration, value_iteration

environment = EasyGridWorld()
GAMMA = 0.9
EPS = 1e-6

policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                      num_actions=environment.num_actions)

print("Policy Evaluation:")
value_function = policy_evaluation(policy, environment, GAMMA, eps=EPS)
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

print("Policy Evaluation:")
value_function = policy_evaluation(policy, environment, GAMMA, eps=EPS,
                                   value_function=value_function)
print(value_function.table)
print()


