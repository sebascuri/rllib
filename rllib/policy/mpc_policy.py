"""Policy that Implements MPC."""

from .abstract_policy import AbstractPolicy
from rllib.algorithms.control import cem_shooting, random_shooting
import torch


class MPCPolicy(AbstractPolicy):
    """MPC Policy."""

    def __init__(self, dynamic_model, reward_model, horizon, terminal_reward=None,
                 termination=None, gamma=1, num_samples=400, num_iter=5,
                 num_elites=None, solver='CEM', warm_start=False):
        super().__init__(dynamic_model.dim_state, dynamic_model.dim_action)
        self.dynamic_model = dynamic_model
        self.reward_model = reward_model
        self.terminal_reward = terminal_reward
        self.termination = termination
        self.horizon = horizon
        self.gamma = gamma

        self.num_samples, self.num_elites = num_samples, num_elites
        self.num_iter = num_iter

        self.solver = solver
        self.last_actions = None
        self.warm_start = warm_start

    def forward(self, state):
        """Solve the MPC problem."""
        if self.solver.lower() == 'cem':
            actions = cem_shooting(
                self.dynamic_model, self.reward_model, self.horizon, state,
                num_samples=self.num_samples, gamma=self.gamma,
                num_iter=self.num_iter, num_elites=self.num_elites,
                termination=self.termination, terminal_reward=self.terminal_reward,
                warm_start=self.last_actions
            )
        elif self.solver.lower() == 'random_shooting':
            actions = random_shooting(
                self.dynamic_model, self.reward_model, self.horizon, state,
                num_samples=self.num_samples, gamma=self.gamma,
                termination=self.termination, terminal_reward=self.terminal_reward,
                warm_start=self.last_actions
            )
        else:
            raise NotImplementedError

        if self.warm_start:  # Warm start
            self.last_actions = torch.cat((actions[..., 1:, :],
                                           torch.zeros_like(actions[..., :1, :])),
                                          dim=-2)
        return actions[..., 0, :], torch.zeros(1)  # Return first Step.

    def reset(self):
        """Re-set last_action to None."""
        self.last_actions = None
