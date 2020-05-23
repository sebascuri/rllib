"""Utilities for Pusher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from exps.gpucrl.util import get_mpc_agent, get_mb_mppo_agent
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState, \
    NextStateClamper
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import PusherReward


class QuaternionTransform(nn.Module):
    """Transform pusher states to quaternion representation."""
    extra_dim = 7

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., :7]
        vel, obj = states[..., 7:14], states[..., 14:17]
        return torch.cat((torch.cos(angles), torch.sin(angles), vel, obj), dim=-1)
        # return states

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin, other = states[..., :7], states[..., 7:14], states[..., 14:]
        angles = torch.atan2(sin, cos)
        return torch.cat((angles, other), dim=-1)


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return (state[..., -3:].abs() > 25).any(-1) | (
            state[..., 7:14].abs() > 2000).any(-1)


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # %% Define Environment.
    environment = GymEnvironment('MBRLPusher-v0', action_cost=params.action_cost,
                                 seed=params.seed)
    action_scale = environment.action_scale
    reward_model = PusherReward(action_cost=params.action_cost)

    # %% Define Helper modules
    low = torch.tensor([
        [-2.2854, -0.5236, -1.5, -2.3213, -1.5, -1.094, -1.5,  # qpos
         -10., -10., -10., -10., -10., -10., -10.,  # qvel
         0.3, -0.7, -0.275]
    ])
    high = torch.tensor([
        [2.0, 1.3963, 1.7, 0, 1.5, 0, 1.5,  # qpos
         10., 10., 10., 10., 10., 10., 10.,  # qvel
         0.8, 0.1, -0.275]
    ])

    transformations = [
        NextStateClamper(low, high),
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),
    ]

    input_transform = QuaternionTransform()
    # input_transform = None
    # 0, 0.5, -1.5, -1.2, 0.7,
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor(
            [-.3, 0.3, -1.5, -1.5, 0.5, -1.094, -1.5,  # qpos
             -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,  # qvel
             0.5, -0.4, -0.323]  # object
        ),
        torch.tensor(
            [0.3, 0.6, -1., -1., 0.9, 0., 1.5,  # qpos
             0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,  # qvel
             0.7, -0.2, -0.275]  # object
        )
    )

    if agent_name == 'mpc':
        agent = get_mpc_agent(environment.name, environment.dim_state,
                              environment.dim_action,
                              params, reward_model,
                              action_scale=action_scale,
                              transformations=transformations,
                              input_transform=input_transform,
                              termination=large_state_termination,
                              initial_distribution=exploratory_distribution)
    elif agent_name == 'mbmppo':
        agent = get_mb_mppo_agent(
            environment.name, environment.dim_state, environment.dim_action,
            params, reward_model,
            input_transform=input_transform,
            action_scale=action_scale,
            transformations=transformations,
            termination=large_state_termination,
            initial_distribution=exploratory_distribution)
    else:
        raise NotImplementedError

    return environment, agent
