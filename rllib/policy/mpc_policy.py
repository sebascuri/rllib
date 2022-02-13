"""Policy that Implements MPC."""

import torch

from rllib.algorithms.mpc.cem_shooting import CEMShooting

from .abstract_policy import AbstractPolicy


class MPCPolicy(AbstractPolicy):
    """MPC Policy.

    Parameters
    ----------
    mpc_solver: MPCSolver.
        Base solver for the MPC algorithm.

    solver_frequency: int
        How often to call the MPC solver.

    """

    def __init__(self, mpc_solver, solver_frequency=1, *args, **kwargs):
        super().__init__(
            dim_state=mpc_solver.dynamical_model.dim_state,
            dim_action=mpc_solver.dynamical_model.dim_action,
            action_scale=mpc_solver.action_scale,
            goal=mpc_solver.reward_model.goal,
        )
        self._steps = 0
        self.solver_frequency = solver_frequency
        self.action_sequence = None
        self.solver = mpc_solver
        if solver_frequency > self.solver.num_model_steps:
            raise ValueError(
                f"""Solver num model steps has to be larger than the solver frequency.
                But num model steps: {self.solver.num_model_steps} and solver_frequency:
                {solver_frequency}."""
            )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractPolicy.default()."""
        return cls(mpc_solver=CEMShooting(), *args, **kwargs)

    def forward(self, state, **kwargs):
        """Solve the MPC problem."""
        if self._steps % self.solver_frequency == 0 or self.action_sequence is None:
            self.action_sequence = self.solver(state)
        else:
            self.solver.initialize_actions(state.shape[:-1])

        action = self.action_sequence[self._steps % self.solver_frequency, ..., :]
        self._steps += 1
        return action, torch.zeros(self.dim_action[0], self.dim_action[0])

    def reset(self):
        """Re-set last_action to None."""
        self._steps = 0
        self.solver.reset()

    def set_goal(self, goal=None):
        """Set goal."""
        self.solver.reward_model.set_goal(goal)
