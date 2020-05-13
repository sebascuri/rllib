"""Utilities for Reacher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from exps.gpucrl.util import large_state_termination, get_mpc_agent, get_mb_mppo_agent
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import ReacherReward


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""
    extra_dim = 0

    def forward(self, states):
        """Transform state before applying function approximation."""
        return states

    def inverse(self, states):
        """Inverse transformation of states."""
        return states


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # %% Define Helper modules
    transformations = [ActionScaler(scale=params.action_scale),
                       MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
                       ]

    input_transform = StateTransform()

    # %% Define Environment.
    environment = GymEnvironment('MBRLReacher3D-v0', action_cost=params.action_cost,
                                 seed=params.seed)
    reward_model = ReacherReward(action_cost=params.action_cost,
                                 goal=environment.goal)

    exploratory_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(environment.dim_state), torch.eye(environment.dim_state)
    )

    if agent_name == 'mpc':
        agent = get_mpc_agent(environment.name, environment.dim_state,
                              environment.dim_action,
                              params, reward_model, transformations, input_transform,
                              termination=large_state_termination,
                              initial_distribution=exploratory_distribution)
    elif agent_name == 'mbmppo':
        agent = get_mb_mppo_agent(
            environment.name, environment.dim_state, environment.dim_action,
            params, reward_model, input_transform=input_transform,
            transformations=transformations,
            termination=large_state_termination,
            initial_distribution=exploratory_distribution)
    else:
        raise NotImplementedError

    return environment, agent
