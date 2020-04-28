"""Utilities for inverted pendulum experiments."""
import math

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gpytorch
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
from torch.optim.rmsprop import RMSprop

from rllib.algorithms.mppo import MBMPPO, train_mppo
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.dataset.datatypes import Observation
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.abstract_model import AbstractModel
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction
from rllib.reward.abstract_reward import AbstractReward
from rllib.reward.utilities import tolerance
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.util.neural_networks.utilities import freeze_parameters

from experiments.gpucrl_inverted_pendulum.plotters import plot_learning_losses, \
    plot_trajectory_states_and_rewards, plot_values_and_policy


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                            dim=-1)
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((angle, angular_velocity), dim=-1)
        return states_


def termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return (torch.any(torch.abs(state) > 15, dim=-1) | torch.any(
        torch.abs(action) > 15, dim=-1))


class PendulumReward(AbstractReward):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost_ratio=0):
        super().__init__()
        self.action_cost_ratio = action_cost_ratio

    def forward(self, state, action):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1., margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost_ratio * (action_tolerance-1)

        cost = state_cost + action_cost

        return cost, torch.zeros(1)


class PendulumModel(AbstractModel):
    """Pendulum Model.

    Torch implementation of a pendulum model using euler forwards integration.
    """

    def __init__(self, mass, length, friction, step_size=1 / 80,
                 noise: MultivariateNormal = None):
        super().__init__(dim_state=2, dim_action=1)
        self.mass = mass
        self.length = length
        self.friction = friction
        self.step_size = step_size
        self.noise = noise

    def forward(self, state, action):
        """Get next-state distribution."""
        # Physical dynamics
        action = torch.clamp(action, -1., 1.)
        mass = self.mass
        gravity = 9.81
        length = self.length
        friction = self.friction
        inertia = mass * length ** 2
        dt = self.step_size

        angle, angular_velocity = torch.split(state, 1, dim=-1)
        for _ in range(1):
            x_ddot = ((gravity / length) * torch.sin(angle)
                      + action * (1 / inertia)
                      - (friction / inertia) * angular_velocity)

            angle = angle + dt * angular_velocity
            angular_velocity = angular_velocity + dt * x_ddot

        next_state = (torch.cat((angle, angular_velocity), dim=-1))

        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix


def solve_mpc(dynamic_model, action_cost_ratio, num_iter, num_sim_steps, batch_size,
              num_gradient_steps, num_trajectories, num_action_samples, num_episodes,
              epsilon, epsilon_mean, epsilon_var, eta, eta_mean, eta_var, lr):
    reward_model = PendulumReward(action_cost_ratio)
    freeze_parameters(dynamic_model)
    value_function = NNValueFunction(dim_state=2, layers=[64, 64], biased_head=False,
                                     input_transform=StateTransform())

    policy = NNPolicy(dim_state=2, dim_action=1, layers=[64, 64], biased_head=False,
                      squashed_output=True, input_transform=StateTransform())

    value_function = torch.jit.script(value_function)
    init_distribution = torch.distributions.Uniform(torch.tensor([-np.pi, -0.05]),
                                                    torch.tensor([np.pi, 0.05]))

    # %% Define MPC solver.
    mppo = MBMPPO(dynamic_model, reward_model, policy, value_function, gamma=0.99,
                  epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
                  eta=eta, eta_mean=eta_mean, eta_var=eta_var,
                  num_action_samples=num_action_samples)

    optimizer = optim.Adam([p for p in mppo.parameters() if p.requires_grad], lr=lr)

    # %% Train Controller
    test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())

    policy_losses, value_losses, kl_div, returns, entropy = [], [], [], [], []

    for _ in range(num_episodes):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches():
            vloss_, ploss_, kl_div_, return_, entropy_, = train_mppo(
                mppo, init_distribution, optimizer,
                num_iter=num_iter, num_trajectories=num_trajectories,
                num_simulation_steps=num_sim_steps,
                num_gradient_steps=num_gradient_steps,
                batch_size=batch_size, num_subsample=1)

        policy_losses += ploss_
        value_losses += vloss_
        returns += return_
        entropy += entropy_
        kl_div += kl_div_

        # %% Test controller on Model.
        with torch.no_grad():
            trajectory = rollout_model(
                mppo.dynamical_model, mppo.reward_model,
                # policy,
                lambda x: (policy(x)[0], torch.zeros(1)),
                initial_state=test_state.unsqueeze(0).unsqueeze(1),
                max_steps=400)

            trajectory = stack_list_of_tuples(trajectory)

        states = trajectory.state[:, 0]
        rewards = trajectory.reward
        plot_trajectory_states_and_rewards(states, rewards)
        plt.show()
        print(f'Model Cumulative reward: {torch.sum(rewards):.2f}')

        bounds = [(-2 * np.pi, 2 * np.pi), (-12, 12)]
        ax_value, ax_policy = plot_values_and_policy(value_function, policy, bounds,
                                                     [200, 200])
        ax_value.plot(states[:, 0, 0], states[:, 0, 1], color='C1')
        ax_value.plot(states[-1, 0, 0], states[-1, 0, 1], 'x', color='C1')
        plt.show()

        # %% Test controller on Environment.
        environment = SystemEnvironment(
            InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                             step_size=1 / 80), reward=reward_model)
        environment.state = test_state.numpy()
        environment.initial_state = lambda: test_state.numpy()
        trajectory = rollout_policy(environment, max_steps=400, render=False,
                                    policy=lambda x: (policy(x)[0], torch.zeros(1)),
                                    # policy=policy
                                    )

        trajectory = stack_list_of_tuples(trajectory[0])
        print(f'Environment Cumulative reward: {torch.sum(trajectory.reward):.2f}')

    # %% Plot returns and losses.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(np.arange(len(returns)), returns)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Returns')
    # ax1.set_ylim([0, 350])

    ax2.plot(np.arange(len(entropy)), entropy)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Entropy')
    # ax2.set_ylim([-5, 2])

    ax3.plot(np.arange(len(kl_div)), kl_div)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('KL')
    plt.show()

    # %% Plot losses.
    plot_learning_losses(policy_losses, value_losses, horizon=20)
    plt.show()

    return torch.sum(trajectory.reward)
