"""MPC Agent Implementation."""

from rllib.algorithms.mpc import CEMShooting

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(self, mpc_solver, *args, **kwargs):
        super().__init__(
            planning_algorithm=mpc_solver,
            dynamical_model=kwargs.pop("dynamical_model", mpc_solver.dynamical_model),
            reward_model=kwargs.pop("reward_model", mpc_solver.reward_model),
            termination_model=kwargs.pop(
                "termination_model", mpc_solver.termination_model
            ),
            *args,
            **kwargs,
        )

    @classmethod
    def default(
        cls, environment, mpc_solver=None, horizon=25, num_iter=5, *args, **kwargs
    ):
        """See `AbstractAgent.default'."""
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()

        if mpc_solver is None:
            mpc_solver = CEMShooting(
                dynamical_model=agent.dynamical_model,
                reward_model=agent.reward_model,
                termination_model=agent.termination_model,
                action_scale=environment.action_scale,
                horizon=horizon,
                gamma=agent.gamma,
                num_iter=num_iter,
            )
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
