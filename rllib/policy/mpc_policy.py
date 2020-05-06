"""Policy that Implements MPC."""

from .abstract_policy import AbstractPolicy
import torch


class MPCPolicy(AbstractPolicy):
    """MPC Policy."""

    def __init__(self, mpc_solver):
        super().__init__(mpc_solver.dynamical_model.dim_state,
                         mpc_solver.dynamical_model.dim_action)

        self.solver = mpc_solver

    def forward(self, state):
        """Solve the MPC problem."""
        action_sequence = self.solver(state)
        return action_sequence[..., 0, :], torch.zeros(1)  # Return first Step.

    def reset(self):
        """Re-set last_action to None."""
        self.solver.reset()
