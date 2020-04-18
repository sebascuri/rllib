import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.jit
import torch.nn as nn
import torch.optim as optim

from rllib.algorithms.mppo import MBMPPO, train_mppo
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.pendulum_model import PendulumModel
from rllib.policy import NNPolicy
from rllib.reward.pendulum_reward import PendulumReward
from rllib.util.plotting import plot_on_grid, plot_learning_losses, \
    plot_values_and_policy
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.value_function import NNValueFunction

torch.manual_seed(0)
np.random.seed(0)

class StateTransform(nn.Module):
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                            dim=-1)
        return states_

# %% Define Policy, Value function, Model.

value_function = NNValueFunction(dim_state=2, layers=[64, 64], biased_head=False,
                                 input_transform=StateTransform())

policy = NNPolicy(dim_state=2, dim_action=1, layers=[64, 64], biased_head=False,
                  squashed_output=True, input_transform=StateTransform())

dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)
init_distribution = torch.distributions.Uniform(
    torch.tensor([-np.pi, 0.0]),
    torch.tensor([np.pi, 0.0])
)

value_function = torch.jit.script(value_function)
policy = torch.jit.script(policy)
dynamic_model = torch.jit.script(dynamic_model)


# %%  Define EnvironmentDefine Environment
reward_model = PendulumReward(action_cost_ratio=0.2)

test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())
environment = SystemEnvironment(
    InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
    reward=reward_model)
environment.state = test_state.numpy()
environment.initial_state = lambda: test_state.numpy()


# %% Initialialize MB-MPPO and Optimizer
mppo = MBMPPO(dynamic_model, reward_model, policy, value_function,
              epsilon=0.1, epsilon_mean=0.01, epsilon_var=0.001, gamma=0.99,
              num_action_samples=15)
optimizer = optim.Adam(mppo.parameters(), lr=5e-4)

# %%  Train Controller
num_iter = 50
num_simulation_steps = 400
batch_size = 100
refresh_interval = 2
num_trajectories = 8  # math.ceil(num_inner_iterations * 100 / num_simulation_steps)
num_subsample = 1

num_episodes = 4
policy_losses, value_losses, policy_returns = [], [], []
for _ in range(num_episodes):
    value_losses_, policy_losses_, policy_returns_, eta_parameters, kl_div = train_mppo(
        mppo, init_distribution, optimizer,
        num_iter=num_iter, num_trajectories=num_trajectories,
        num_simulation_steps=num_simulation_steps, refresh_interval=refresh_interval,
        batch_size=batch_size, num_subsample=num_subsample)

    policy_losses += policy_losses_
    value_losses += value_losses_
    policy_returns += policy_returns_

    # %% Test controller on Model.
    with torch.no_grad():
        trajectory = rollout_model(mppo.dynamical_model, mppo.reward_model, policy,
                                   initial_state=test_state,
                                   max_steps=400)

        trajectory = Observation(*stack_list_of_tuples(trajectory))

    states = trajectory.state
    rewards = trajectory.reward
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))

    plt.sca(ax1)
    plt.plot(states[:, 0], states[:, 1], 'x')
    plt.plot(states[-1, 0], states[-1, 1], 'x')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Angular velocity [rad/s]')

    plt.sca(ax2)
    plt.plot(rewards)
    plt.xlabel('Time step')
    plt.ylabel('Instantaneous reward')
    plt.show()
    print(f'System Cumulative Reward: {torch.sum(rewards):.2f}')

    bounds = [(-2 * np.pi, 2 * np.pi), (-12, 12)]
    ax_value, ax_policy = plot_values_and_policy(value_function, policy, bounds,
                                                 [200, 200])
    ax_value.plot(states[:, 0], states[:, 1], color='C1')
    ax_value.plot(states[-1, 0], states[-1, 1], 'x', color='C1')
    plt.show()

    # %% Test controller on Environment.
    policy.deterministic = True
    trajectory = rollout_policy(environment, policy, max_steps=400, render=True)
    trajectory = Observation(*stack_list_of_tuples(trajectory[0]))
    print(f'Environment Cumulative reward: {torch.sum(trajectory.reward):.2f}')

    policy.deterministic = False


plt.plot(refresh_interval * np.arange(len(policy_returns)), policy_returns)
plt.xlabel('Iteration')
plt.ylabel('Cumulative reward')
plt.show()

plot_learning_losses(policy_losses, value_losses, horizon=20)
plt.show()