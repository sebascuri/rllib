"""Utilities for inverted pendulum experiments."""

import numpy as np
import gpytorch
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.optim as optim

from rllib.algorithms.mppo import MBMPPO, train_mppo
from rllib.dataset.utilities import stack_list_of_tuples
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
    plot_trajectory_states_and_rewards, plot_values_and_policy, plot_returns_entropy_kl


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

    def __init__(self, action_cost=0):
        super().__init__()
        self.action_cost = action_cost

    def forward(self, state, action, next_state):
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
        action_cost = self.action_cost * (action_tolerance - 1)

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


def test_policy_on_model(dynamical_model, reward_model, policy, test_state,
                         policy_str='Sampled Policy'):
    with torch.no_grad():
        trajectory = rollout_model(dynamical_model, reward_model, policy, max_steps=400,
                                   initial_state=test_state.unsqueeze(0).unsqueeze(1))
        trajectory = stack_list_of_tuples(trajectory)

    states = trajectory.state[:, 0]
    rewards = trajectory.reward
    plot_trajectory_states_and_rewards(states, rewards)

    model_rewards = torch.sum(rewards)
    print(f"Model with {policy_str} Cumulative reward: {model_rewards:.2f}")

    return model_rewards, trajectory


def test_policy_on_environment(environment, policy, test_state,
                               policy_str='Sampled Policy'):
    environment.state = test_state.numpy()
    environment.initial_state = lambda: test_state.numpy()
    trajectory = rollout_policy(environment, policy, max_steps=400, render=False)[0]

    trajectory = stack_list_of_tuples(trajectory)
    env_rewards = torch.sum(trajectory.reward)
    print(f"Environment with {policy_str} Cumulative reward: {env_rewards:.2f}")

    return env_rewards, trajectory


def solve_mpc(dynamical_model, action_cost, num_iter, num_sim_steps, batch_size,
              num_gradient_steps, num_trajectories, num_action_samples, num_episodes,
              epsilon, epsilon_mean, epsilon_var, eta, eta_mean, eta_var, lr):
    reward_model = PendulumReward(action_cost)
    freeze_parameters(dynamical_model)
    value_function = NNValueFunction(dim_state=2, layers=[64, 64], biased_head=False,
                                     input_transform=StateTransform())

    policy = NNPolicy(dim_state=2, dim_action=1, layers=[64, 64], biased_head=False,
                      squashed_output=True, input_transform=StateTransform())

    value_function = torch.jit.script(value_function)
    init_distribution = torch.distributions.Uniform(torch.tensor([-np.pi, -0.05]),
                                                    torch.tensor([np.pi, 0.05]))

    # %% Define MPC solver.
    mppo = MBMPPO(dynamical_model, reward_model, policy, value_function, gamma=0.99,
                  epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
                  eta=eta, eta_mean=eta_mean, eta_var=eta_var,
                  num_action_samples=num_action_samples)

    optimizer = optim.Adam([p for p in mppo.parameters() if p.requires_grad], lr=lr)

    # %% Train Controller
    test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())

    policy_losses, value_losses, kl_div, returns, entropy = [], [], [], [], []
    environment_rewards, trajectory = 0, None

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
        test_policy_on_model(mppo.dynamical_model, mppo.reward_model, mppo.policy,
                             test_state)
        _, trajectory = test_policy_on_model(
            mppo.dynamical_model, mppo.reward_model,
            lambda x: (mppo.policy(x)[0][:mppo.dynamical_model.dim_action],
                       torch.zeros(1)),
            test_state, policy_str='Expected Policy')

        # %% Test controller on Environment.
        environment = SystemEnvironment(
            InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                             step_size=1 / 80), reward=reward_model)
        test_policy_on_environment(environment, mppo.policy, test_state)

        environment_rewards, _ = test_policy_on_environment(
            environment,
            lambda x: (mppo.policy(x)[0][:mppo.dynamical_model.dim_action],
                       torch.zeros(1)),
            test_state, policy_str='Expected Policy')

    # %% Plots
    # Plot value funciton and policy.
    plot_values_and_policy(value_function, policy, trajectory=trajectory,
                           num_entries=[200, 200],
                           bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)])

    # Plot returns and losses.
    plot_returns_entropy_kl(returns, entropy, kl_div)

    # Plot losses.
    plot_learning_losses(policy_losses, value_losses, horizon=20)

    return environment_rewards
