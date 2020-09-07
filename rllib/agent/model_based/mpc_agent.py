"""MPC Agent Implementation."""

from rllib.algorithms.mpc import CEMShooting

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(self, mpc_solver, *args, **kwargs):
        super().__init__(planning_algorithm=mpc_solver, *args, **kwargs)

    @classmethod
    def default(cls, environment, mpc_solver=None, *args, **kwargs):
        """See `AbstractAgent.default'."""
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()

        if mpc_solver is None:
            mpc_solver = CEMShooting(
                dynamical_model=agent.dynamical_model,
                reward_model=agent.reward_model,
                termination_model=agent.termination_model,
                horizon=5 if kwargs.get("test", False) else 25,
                gamma=agent.gamma,
                num_iter=2 if kwargs.get("test", False) else 5,
                num_samples=20 if kwargs.get("test", False) else 400,
                num_elites=5 if kwargs.get("test", False) else 40,
            )
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
