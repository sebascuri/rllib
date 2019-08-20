from rllib.environment.mdp import EasyGridWorld
from rllib.policy import RandomPolicy
from rllib.algorithms import policy_evaluation, policy_iteration, value_iteration

environment = EasyGridWorld()
GAMMA = 0.9

policy = RandomPolicy(dim_state=1, dim_action=1, num_states=environment.num_states,
                      num_actions=environment.num_actions)

print("Policy Evaluation:")
value_function = policy_evaluation(policy, environment, GAMMA)
print(value_function.parameters)

print("Policy Iteration:")
policy, value_function = policy_iteration(environment, GAMMA)
print(policy.parameters.argmax(dim=1))
print(value_function.parameters)

print("Value Iteration")
policy, value_function = value_iteration(environment, GAMMA)
print(policy.parameters.argmax(dim=1))
print(value_function.parameters)

print("Policy Evaluation:")
value_function = policy_evaluation(policy, environment, GAMMA,
                                   value_function=value_function)
print(value_function.parameters)


