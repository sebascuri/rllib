"""Utilities for Reacher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from exps.gpucrl.util import get_mpc_agent, get_mb_mppo_agent
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import ReacherReward


class QuaternionTransform(nn.Module):
    """Transform reacher states to quaternion representation."""
    extra_dim = 7

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles, other = states[..., :7], states[..., 7:]
        return torch.cat((torch.cos(angles), torch.sin(angles), other), dim=-1)

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
        torch.abs(action) > 25 * 20, dim=-1))


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # %% Define Environment.
    environment = GymEnvironment('MBRLReacher3D-v0', action_cost=params.action_cost,
                                 seed=params.seed)
    action_scale = environment.action_scale
    reward_model = ReacherReward(action_cost=params.action_cost)

    # %% Define Helper modules
    transformations = [ActionScaler(scale=action_scale),
                       MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
                       ]

    input_transform = QuaternionTransform()
    # input_transform = None
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor(
            [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi,  # qpos
             -0.35, -0.1, -0.35,  # goal
             -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,  # qvel
             ]),
        torch.tensor(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi,  # qpos
             0.35, 0.6, 0.35,  # goal
             0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,  # qvel
             ])
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
