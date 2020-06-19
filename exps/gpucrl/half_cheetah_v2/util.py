"""Utilities for Half-Cheetah experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from exps.gpucrl.util import get_mb_mppo_agent, get_mpc_agent
from rllib.dataset.transforms import ActionScaler, MeanFunction
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import HalfCheetahV2Reward


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 8

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., 1:9]
        states_ = torch.cat(
            (states[..., :1], torch.cos(angles), torch.sin(angles), states[..., 9:]),
            dim=-1,
        )
        return states_

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin = states[..., 2:9], states[..., 9:16]
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((states[..., :2], angle, states[..., 16:]), dim=-1)
        return states_


class CheetahMeanFunction(nn.Module):
    """Implementation of a Mean function that returns f(s, a) = [0, s_1:N]."""

    def forward(self, state, action):
        """Compute next state."""
        return torch.cat((torch.zeros_like(state)[..., 0], state[..., 1:]))


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return torch.any(torch.abs(state[..., 1:]) > 2000, dim=-1) | torch.any(
        torch.abs(action) > 25 * 4, dim=-1
    )


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    environment = GymEnvironment(
        "MBRLHalfCheetah-v2", action_cost=params.action_cost, seed=params.seed
    )
    action_scale = environment.action_scale
    reward_model = HalfCheetahV2Reward(action_cost=params.action_cost)

    # %% Define Helper modules
    transformations = [
        ActionScaler(scale=action_scale),
        MeanFunction(CheetahMeanFunction()),  # AngleWrapper(indexes=[1])
    ]

    input_transform = None
    exploratory_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(environment.dim_state), 1e-6 * torch.eye(environment.dim_state)
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
    else:
        raise NotImplementedError

    return environment, agent