"""MPC Agent Implementation."""
import importlib

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
        cls,
        environment,
        mpc_solver=None,
        mpc_solver_name="CEMShooting",
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()
        kwargs.update(
            dynamical_model=agent.dynamical_model,
            reward_model=agent.reward_model,
            termination_model=agent.termination_model,
            gamma=agent.gamma,
        )
        if mpc_solver is None:
            solver_module = importlib.import_module("rllib.algorithms.mpc")
            mpc_solver = getattr(solver_module, mpc_solver_name)(
                action_scale=environment.action_scale, *args, **kwargs
            )
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
