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
        # Return first Step.
        return action_sequence[0, ..., :], torch.zeros(self.dim_action, self.dim_action)

    def reset(self):
        """Re-set last_action to None."""
        self.solver.reset()
