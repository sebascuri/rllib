"""Utilities for inverted pendulum experiments."""
from typing import List

import gpytorch
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from exps.gpucrl.inverted_pendulum.plotters import (
    plot_learning_losses,
    plot_returns_entropy_kl,
    plot_trajectory_states_and_rewards,
    plot_values_and_policy,
)
from exps.gpucrl.util import get_mb_mppo_agent, get_mb_sac_agent, get_mpc_agent
from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.abstract_model import AbstractModel
from rllib.policy import NNPolicy
from rllib.reward.abstract_reward import AbstractReward
from rllib.reward.utilities import tolerance
from rllib.util.neural_networks.utilities import freeze_parameters
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.value_function import NNValueFunction


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1
        )
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((angle, angular_velocity), dim=-1)
        return states_


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return torch.any(torch.abs(state) > 200, dim=-1) | torch.any(
        torch.abs(action) > 200, dim=-1
    )


class PendulumReward(AbstractReward):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__()
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost, torch.zeros(1)


class PendulumModel(AbstractModel):
    """Pendulum Model.

    Torch implementation of a pendulum model using euler forwards integration.
    """

    def __init__(
        self, mass, length, friction, step_size=1 / 80, noise: MultivariateNormal = None
    ):
        super().__init__(dim_state=2, dim_action=1)
        self.mass = mass
        self.length = length
        self.friction = friction
        self.step_size = step_size
        self.noise = noise

    def forward(self, state, action):
        """Get next-state distribution."""
        # Physical dynamics
        action = torch.clamp(action, -1.0, 1.0)
        mass = self.mass
        gravity = 9.81
        length = self.length
        friction = self.friction
        inertia = mass * length ** 2
        dt = self.step_size

        angle, angular_velocity = torch.split(state, 1, dim=-1)
        for _ in range(1):
            x_ddot = (
                (gravity / length) * torch.sin(angle)
                + action * (1 / inertia)
                - (friction / inertia) * angular_velocity
            )

            angle = angle + dt * angular_velocity
            angular_velocity = angular_velocity + dt * x_ddot

        next_state = torch.cat((angle, angular_velocity), dim=-1)

        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix


def test_policy_on_model(
    dynamical_model, reward_model, policy, test_state, policy_str="Sampled Policy"
):
    """Test a policy on a model."""
    with torch.no_grad():
        trajectory = rollout_model(
            dynamical_model,
            reward_model,
            policy,
            max_steps=400,
            initial_state=test_state.unsqueeze(0).unsqueeze(1),
        )
        trajectory = stack_list_of_tuples(trajectory)

    states = trajectory.state[:, 0]
    rewards = trajectory.reward
    plot_trajectory_states_and_rewards(states, rewards)

    model_rewards = torch.sum(rewards).item()
    print(f"Model with {policy_str} Cumulative reward: {model_rewards:.2f}")

    return model_rewards, trajectory


def test_policy_on_environment(
    environment, policy, test_state, policy_str="Sampled Policy"
):
    """Test a policy on an environment."""
    environment.state = test_state.numpy()
    environment.initial_state = lambda: test_state.numpy()
    trajectory = rollout_policy(environment, policy, max_steps=400, render=False)[0]

    trajectory = stack_list_of_tuples(trajectory)
    env_rewards = torch.sum(trajectory.reward).item()
    print(f"Environment with {policy_str} Cumulative reward: {env_rewards:.2f}")

    return env_rewards, trajectory


def train_mppo(
    mppo: MBMPPO,
    initial_distribution,
    optimizer,
    num_iter,
    num_trajectories,
    num_simulation_steps,
    num_gradient_steps,
    batch_size,
    num_subsample,
):
    """Train MPPO policy."""
    value_losses = []  # type: List[float]
    policy_losses = []  # type: List[float]
    returns = []  # type: List[float]
    kl_div = []  # type: List[float]
    entropy = []  # type: List[float]
    for i in tqdm(range(num_iter)):
        # Compute the state distribution
        state_batches = _simulate_model(
            mppo,
            initial_distribution,
            num_trajectories,
            num_simulation_steps,
            batch_size,
            num_subsample,
            returns,
            entropy,
        )

        policy_episode_loss, value_episode_loss, episode_kl_div = _optimize_policy(
            mppo, state_batches, optimizer, num_gradient_steps
        )

        value_losses.append(value_episode_loss / len(state_batches))
        policy_losses.append(policy_episode_loss / len(state_batches))
        kl_div.append(episode_kl_div)

    return value_losses, policy_losses, kl_div, returns, entropy


def _simulate_model(
    mppo,
    initial_distribution,
    num_trajectories,
    num_simulation_steps,
    batch_size,
    num_subsample,
    returns,
    entropy,
):
    with torch.no_grad():
        test_states = torch.tensor([np.pi, 0]).repeat(num_trajectories // 2, 1)
        initial_states = initial_distribution.sample((num_trajectories // 2,))
        initial_states = torch.cat((initial_states, test_states), dim=0)
        trajectory = rollout_model(
            mppo.dynamical_model,
            reward_model=mppo.reward_model,
            policy=mppo.policy,
            initial_state=initial_states,
            max_steps=num_simulation_steps,
        )
        trajectory = stack_list_of_tuples(trajectory)
        returns.append(trajectory.reward.sum(dim=0).mean().item())
        entropy.append(trajectory.entropy.mean())
        # Shuffle to get a state distribution
        states = trajectory.state.reshape(-1, trajectory.state.shape[-1])
        np.random.shuffle(states.numpy())
        state_batches = torch.split(states, batch_size)[::num_subsample]

    return state_batches


def _optimize_policy(mppo, state_batches, optimizer, num_gradient_steps):
    policy_episode_loss = 0.0
    value_episode_loss = 0.0
    episode_kl_div = 0.0

    # Copy over old policy for KL divergence
    mppo.reset()

    # Iterate over state batches in the state distribution
    for _ in range(num_gradient_steps):
        idx = np.random.choice(len(state_batches))
        states = state_batches[idx]
        optimizer.zero_grad()
        losses = mppo(states)
        losses.loss.backward()
        optimizer.step()

        # Track statistics
        value_episode_loss += losses.critic_loss.item()
        policy_episode_loss += losses.policy_loss.item()
        # episode_kl_div += losses.kl_div.item()
        mppo.update()

    return policy_episode_loss, value_episode_loss, episode_kl_div


def solve_mppo(
    dynamical_model,
    action_cost,
    num_iter,
    num_sim_steps,
    batch_size,
    num_gradient_steps,
    num_trajectories,
    num_action_samples,
    num_episodes,
    epsilon,
    epsilon_mean,
    epsilon_var,
    eta,
    eta_mean,
    eta_var,
    lr,
):
    """Solve MPPO optimization problem."""
    reward_model = PendulumReward(action_cost)
    freeze_parameters(dynamical_model)
    value_function = NNValueFunction(
        dim_state=2,
        layers=[64, 64],
        biased_head=False,
        input_transform=StateTransform(),
    )

    policy = NNPolicy(
        dim_state=2,
        dim_action=1,
        layers=[64, 64],
        biased_head=False,
        squashed_output=True,
        input_transform=StateTransform(),
    )

    # value_function = torch.jit.script(value_function)
    init_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.05]), torch.tensor([np.pi, 0.05])
    )

    # %% Define MPC solver.
    mppo = MBMPPO(
        dynamical_model,
        reward_model,
        policy,
        value_function,
        gamma=0.99,
        epsilon=epsilon,
        epsilon_mean=epsilon_mean,
        epsilon_var=epsilon_var,
        eta=eta,
        eta_mean=eta_mean,
        eta_var=eta_var,
        num_action_samples=num_action_samples,
        criterion=nn.MSELoss,
    )

    optimizer = optim.Adam([p for p in mppo.parameters() if p.requires_grad], lr=lr)

    # %% Train Controller
    test_state = torch.tensor(np.array([np.pi, 0.0]), dtype=torch.get_default_dtype())

    policy_losses, value_losses, kl_div, returns, entropy = [], [], [], [], []
    model_rewards, trajectory = 0, None

    for _ in range(num_episodes):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches():
            vloss_, ploss_, kl_div_, return_, entropy_, = train_mppo(
                mppo,
                init_distribution,
                optimizer,
                num_iter=num_iter,
                num_trajectories=num_trajectories,
                num_simulation_steps=num_sim_steps,
                num_gradient_steps=num_gradient_steps,
                batch_size=batch_size,
                num_subsample=1,
            )

        policy_losses += ploss_
        value_losses += vloss_
        returns += return_
        entropy += entropy_
        kl_div += kl_div_

        # # %% Test controller on Model.
        test_policy_on_model(
            mppo.dynamical_model, mppo.reward_model, mppo.policy, test_state
        )
        _, trajectory = test_policy_on_model(
            mppo.dynamical_model,
            mppo.reward_model,
            lambda x: (
                mppo.policy(x)[0][: mppo.dynamical_model.dim_action],
                torch.zeros(1),
            ),
            test_state,
            policy_str="Expected Policy",
        )

        model_rewards, _ = test_policy_on_model(
            mppo.dynamical_model, mppo.reward_model, mppo.policy, test_state
        )

        # %% Test controller on Environment.
        environment = SystemEnvironment(
            # ModelSystem(PendulumModel(mass=0.3, length=0.5, friction=0.005)),
            InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
            reward=reward_model,
        )
        environment_rewards, trajectory = test_policy_on_environment(
            environment, mppo.policy, test_state
        )

        environment_rewards, _ = test_policy_on_environment(
            environment,
            lambda x: (
                mppo.policy(x)[0][: mppo.dynamical_model.dim_action],
                torch.zeros(1),
            ),
            test_state,
            policy_str="Expected Policy",
        )

    # %% Plots
    # Plot value funciton and policy.
    plot_values_and_policy(
        value_function,
        policy,
        trajectory=trajectory,
        num_entries=[200, 200],
        bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)],
    )

    # Plot returns and losses.
    plot_returns_entropy_kl(returns, entropy, kl_div)

    # Plot losses.
    plot_learning_losses(policy_losses, value_losses, horizon=20)

    return model_rewards


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    initial_distribution = torch.distributions.Uniform(
        torch.tensor([np.pi, -0.0]), torch.tensor([np.pi, +0.0])
    )
    reward_model = PendulumReward(action_cost=params.action_cost)
    environment = SystemEnvironment(
        InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
        reward=reward_model,
        initial_state=initial_distribution.sample,
        termination=large_state_termination,
    )

    action_scale = environment.action_scale

    # %% Define Helper modules
    transformations = [
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
    ]

    input_transform = StateTransform()
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])
    )

    if agent_name == "mpc":
        agent = get_mpc_agent(
            environment.dim_state,
            environment.dim_action,
            params,
            reward_model,
            action_scale=action_scale,
            transformations=transformations,
            input_transform=input_transform,
            termination=large_state_termination,
            initial_distribution=exploratory_distribution,
        )
    elif agent_name == "mbmppo":
        agent = get_mb_mppo_agent(
            environment.dim_state,
            environment.dim_action,
            params,
            reward_model,
            input_transform=input_transform,
            action_scale=action_scale,
            transformations=transformations,
            termination=large_state_termination,
            initial_distribution=exploratory_distribution,
        )

    elif agent_name == "mbsac":
        agent = get_mb_sac_agent(
            environment.dim_state,
            environment.dim_action,
            params,
            reward_model,
            input_transform=input_transform,
            action_scale=action_scale,
            transformations=transformations,
            termination=large_state_termination,
            initial_distribution=exploratory_distribution,
        )

    else:
        raise NotImplementedError

    return environment, agent
