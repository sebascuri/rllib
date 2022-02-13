"""MPC Algorithms."""
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.multi_objective_reduction import MeanMultiObjectiveReduction
from rllib.util.neural_networks.utilities import repeat_along_dimension, to_torch
from rllib.util.rollout import rollout_actions
from rllib.util.value_estimation import discount_sum


class MPCSolver(nn.Module, metaclass=ABCMeta):
    r"""Solve the discrete time trajectory optimization controller.

    ..math :: u[0:H-1] = \arg \max \sum_{t=0}^{H-1} r(x0, u) + final_reward(x_H)

    When called, it will return the sequence of actions that solves the problem.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    num_model_steps: int.
        Horizon to solve planning problem.
    gamma: float, optional.
        Discount factor.
    scale: float, optional.
        Scale of covariance matrix to sample.
    num_iter: int, optional.
        Number of iterations of solver method.
    num_particles: int, optional.
        Number of particles for shooting method.
    termination_model: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.
    warm_start: bool, optional.
        Whether or not to start the optimization with a warm start.
    default_action: str, optional.
         Default action behavior.
    num_cpu: int, optional.
        Number of CPUs to run the solver.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_model_steps=25,
        gamma=1.0,
        num_iter=1,
        num_particles=400,
        termination_model=None,
        scale=0.3,
        terminal_reward=None,
        warm_start=True,
        clamp=True,
        default_action="zero",
        action_scale=1.0,
        num_cpu=1,
        multi_objective_reduction=MeanMultiObjectiveReduction(dim=-1),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model

        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.num_model_steps = num_model_steps
        self.gamma = gamma

        self.num_iter = num_iter
        self.num_particles = num_particles
        self.terminal_reward = terminal_reward
        self.warm_start = warm_start
        self.default_action = default_action
        self.dim_action = self.dynamical_model.dim_action[0]
        self.dim_reward = self.reward_model.dim_reward[0]

        self.mean = None
        self._scale = scale
        self.covariance = (scale ** 2) * torch.eye(self.dim_action).repeat(
            self.num_model_steps, 1, 1
        )
        if isinstance(action_scale, np.ndarray):
            action_scale = to_torch(action_scale)
        elif not isinstance(action_scale, torch.Tensor):
            action_scale = torch.full((self.dim_action,), action_scale)
        if len(action_scale) < self.dim_action:
            extra_dim = self.dim_action - len(action_scale)
            action_scale = torch.cat((action_scale, torch.ones(extra_dim)))

        self.action_scale = action_scale
        self.clamp = clamp
        self.num_cpu = num_cpu
        self.multi_objective_reduction = multi_objective_reduction

    def evaluate_action_sequence(self, action_sequence, state):
        """Evaluate action sequence by performing a rollout."""
        trajectory = stack_list_of_tuples(
            rollout_actions(
                self.dynamical_model,
                self.reward_model,
                self.action_scale * action_sequence,  # scale actions.
                state,
                self.termination_model,
            ),
            dim=-2,
        )

        returns = discount_sum(trajectory.reward, self.gamma)

        if self.terminal_reward:
            terminal_reward = self.terminal_reward(trajectory.next_state[..., -1, :])
            returns = returns + self.gamma ** self.num_model_steps * terminal_reward
        return returns

    @abstractmethod
    def get_candidate_action_sequence(self):
        """Get candidate actions."""
        raise NotImplementedError

    @abstractmethod
    def get_best_action(self, action_sequence, returns):
        """Get best action."""
        raise NotImplementedError

    @abstractmethod
    def update_sequence_generation(self, elite_actions):
        """Update sequence generation."""
        raise NotImplementedError

    def initialize_actions(self, batch_shape):
        """Initialize mean and covariance of action distribution."""
        if self.warm_start and self.mean is not None:
            next_mean = self.mean[1:, ..., :]
            if self.default_action == "zero":
                final_action = torch.zeros_like(self.mean[:1, ..., :])
            elif self.default_action == "constant":
                final_action = self.mean[-1:, ..., :]
            elif self.default_action == "mean":
                final_action = torch.mean(next_mean, dim=0, keepdim=True)
            else:
                raise NotImplementedError
            self.mean = torch.cat((next_mean, final_action), dim=0)
        else:
            self.mean = torch.zeros(self.num_model_steps, *batch_shape, self.dim_action)
        self.covariance = (self._scale ** 2) * torch.eye(self.dim_action).repeat(
            self.num_model_steps, *batch_shape, 1, 1
        )

    def forward(self, state):
        """Return action that solves the MPC problem."""
        self.dynamical_model.eval()
        batch_shape = state.shape[:-1]
        self.initialize_actions(batch_shape)
        state = repeat_along_dimension(state, number=self.num_particles, dim=-2)

        for _ in range(self.num_iter):
            action_sequence = self.get_candidate_action_sequence()
            returns = self.evaluate_action_sequence(action_sequence, state)
            elite_actions = self.get_best_action(action_sequence, returns)
            self.update_sequence_generation(elite_actions)

        if self.clamp:
            return self.mean.clamp(-1.0, 1.0)

        return self.mean

    def reset(self, warm_action=None):
        """Reset warm action."""
        self.mean = warm_action
