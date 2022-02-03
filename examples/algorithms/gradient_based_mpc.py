"""Example of gradient based mpc."""
import numpy as np
import torch
from pendulum_utilities import PendulumDenseReward, PendulumModel, PendulumSparseReward

from rllib.algorithms.mpc.gradient_based_solver import GradientBasedSolver
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.policy.mpc_policy import MPCPolicy
from rllib.util.rollout import rollout_policy

action_cost = 0.1
sparse_reward = False
sparse_reward_model = PendulumSparseReward(action_cost=action_cost)
dense_reward_model = PendulumDenseReward(action_cost=action_cost)
reward_model = sparse_reward_model if sparse_reward else dense_reward_model
dynamical_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

eps = 0.001
initial_distribution = torch.distributions.Uniform(
    torch.tensor([np.pi - eps, -eps]), torch.tensor([np.pi + eps, eps])
)
environment = SystemEnvironment(
    InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
    reward=reward_model,
    initial_state=initial_distribution.sample,
)

# dynamical_model = TransformedModel.default(environment)

solver = GradientBasedSolver(
    dim_state=environment.dim_state,
    dim_action=environment.dim_action,
    horizon=10,
    dynamical_model=dynamical_model,
    reward_model=reward_model,
)
policy = MPCPolicy(mpc_solver=solver)
trajectory = rollout_policy(
    policy=policy, environment=environment, max_steps=400, num_episodes=1, render=False
)
print(stack_list_of_tuples(trajectory[0], dim=-2).reward.sum())
