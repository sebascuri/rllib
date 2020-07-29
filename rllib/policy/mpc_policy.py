"""Policy that Implements MPC."""

import torch

from .abstract_policy import AbstractPolicy


class MPCPolicy(AbstractPolicy):
    """MPC Policy."""

    def __init__(self, mpc_solver):
        super().__init__(
            mpc_solver.dynamical_model.dim_state,
            mpc_solver.dynamical_model.dim_action,
            action_scale=mpc_solver.action_scale,
            goal=mpc_solver.reward_model.goal,
        )

        self.solver = mpc_solver

    def forward(self, state, **kwargs):
        """Solve the MPC problem."""
        action_sequence = self.solver(state)
        # Return first Step.
        return (
            action_sequence[0, ..., :],
            torch.zeros(self.dim_action[0], self.dim_action[0]),
        )

    def reset(self):
        """Re-set last_action to None."""
        self.solver.reset()

    def set_goal(self, goal=None):
        """Set goal."""
        self.solver.reward_model.set_goal(goal)
