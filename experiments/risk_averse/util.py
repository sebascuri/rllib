"""Utilities for risk averse experiments."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt

from rllib.reward import AbstractReward
from rllib.dataset.utilities import stack_list_of_tuples


class Cart1dReward(AbstractReward):
    """Reward class for Cart-1D experiment."""

    def __init__(self, goal_x=1., reward_goal=50., reward_step=-1.,
                 high_v=1., reward_high_v=-10., prob_reward_high_v=0.):
        super().__init__()
        self.goal_x = goal_x
        dtype = torch.get_default_dtype()
        self.reward_goal = torch.tensor(reward_goal, dtype=dtype)
        self.reward_step = torch.tensor(reward_step, dtype=dtype)
        self.high_v = high_v
        self.reward_high_v = torch.tensor(reward_high_v, dtype=dtype)
        self.penal_v_dist = Bernoulli(probs=prob_reward_high_v)

    def forward(self, state, action):
        """Compute Reward"""
        x, v = state[..., 0], state[..., 1]
        if x >= self.goal_x:
            return self.reward_goal, torch.zeros(1)
        else:
            reward = self.reward_step
            if np.abs(v) > self.high_v:
                reward = reward + self.reward_high_v * self.penal_v_dist.sample()
            return reward, torch.zeros(1)


class Cart1dTermination(nn.Module):
    def __init__(self, goal_x=1.0):
        super().__init__()
        self.goal_x = goal_x

    def forward(self, state, action, next_state=None):
        x, v = state[..., 0], state[..., 1]
        return x > self.goal_x


def plot_cart_trajectories(agent, episode: int):
    trajectory = stack_list_of_tuples(agent.last_trajectory)

    fig, axes = plt.subplots(4, 1, sharex='row')
    axes[0].plot(trajectory.state[..., 0].numpy())
    axes[1].plot(trajectory.state[..., 1].numpy())
    axes[2].plot(trajectory.action[..., 0].numpy())
    axes[3].plot(trajectory.reward.numpy())

    axes[0].set_ylabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[2].set_ylabel('Acceleration')
    axes[3].set_ylabel('Reward')

    axes[-1].set_xlabel('Time Steps.')
    axes[0].set_title(f"Episode {episode + 1}")

    plt.show()
    plt.close()
