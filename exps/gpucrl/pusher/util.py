"""Utilities for Pusher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from exps.gpucrl.util import get_mpc_agent, get_mb_mppo_agent
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import PusherReward


class QuaternionTransform(nn.Module):
    """Transform pusher states to quaternion representation."""
    extra_dim = 7

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., :7]
        vel, ee, obj = states[..., 7:14], states[..., 14:17], states[..., 17:20]
        return torch.cat((torch.cos(angles), torch.sin(angles), vel, ee, obj), dim=-1)

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

    return (torch.any(torch.abs(state) > 2000, dim=-1) | torch.any(
        torch.abs(action) > 25 * 2, dim=-1))


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
    transformations = [ActionScaler(scale=action_scale),
                       MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
                       ]

    input_transform = QuaternionTransform()
    # input_transform = None
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor(
            [0., 0., 0., 0., 0., 0., 0.,  # qpos
             -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,  # qvel
             0.821, -0.6, 0.,  # tips_arm
             0.5, -0.30655662,  # object
             -0.275]  # table
        ),
        torch.tensor(
            [0., 0., 0., 0., 0., 0., 0.,  # qpos
             0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,  # qvel
             0.821, -0.4, 0.,  # tips_arm
             0.7, -0.30655662,  # object
             -0.275]  # table
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
