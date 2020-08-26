"""MPC Agent Implementation."""

import torch
from torch.optim import Adam

from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc import CEMShooting
from rllib.model import EnsembleModel, TransformedModel
from rllib.reward.quadratic_reward import QuadraticReward

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(self, mpc_solver, *args, **kwargs):
        super().__init__(planning_algorithm=mpc_solver, *args, **kwargs)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        model = EnsembleModel(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_heads=5,
            layers=[200, 200],
            biased_head=False,
            non_linearity="ReLU",
            input_transform=None,
            deterministic=False,
        )
        dynamical_model = TransformedModel(model, kwargs.get("transformations", list()))
        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        reward_model = kwargs.get(
            "reward_model",
            QuadraticReward(
                torch.eye(environment.dim_state[0]),
                torch.eye(environment.dim_action[0]),
                goal=environment.goal,
            ),
        )

        mpc_solver = CEMShooting(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=5 if kwargs.get("test", False) else 25,
            gamma=kwargs.get("gamma", 1.0),
            num_iter=2 if kwargs.get("test", False) else 5,
            num_samples=20 if kwargs.get("test", False) else 400,
            num_elites=5 if kwargs.get("test", False) else 40,
            termination=None,
            terminal_reward=None,
            warm_start=True,
            default_action="zero",
            num_cpu=1,
        )
        model_learning_algorithm = ModelLearningAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_epochs=4 if kwargs.get("test", False) else 30,
            batch_size=64,
            bootstrap=True,
            model_optimizer=model_optimizer,
        )
        return cls(
            mpc_solver=mpc_solver,
            model_learning_algorithm=model_learning_algorithm,
            comment=environment.name,
            *args,
            **kwargs,
        )
