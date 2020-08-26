"""Derived Agent."""

import torch
from torch.optim import Adam

from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc.policy_shooting import PolicyShooting
from rllib.algorithms.simulation_algorithm import SimulationAlgorithm
from rllib.model import EnsembleModel, TransformedModel
from rllib.reward.quadratic_reward import QuadraticReward

from .model_based_agent import ModelBasedAgent


class DerivedMBAgent(ModelBasedAgent):
    """Implementation of a Derived Agent.

    A Derived Agent gets a model-free algorithm and uses the model to derive an
    algorithm.
    """

    def __init__(
        self,
        base_algorithm,
        derived_algorithm_,
        dynamical_model,
        reward_model,
        num_samples=15,
        num_steps=1,
        termination=None,
        *args,
        **kwargs,
    ):
        algorithm = derived_algorithm_(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination=termination,
            num_steps=num_steps,
            num_samples=num_samples,
            *args,
            **kwargs,
        )

        super().__init__(policy_learning_algorithm=algorithm, *args, **kwargs)

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.policy_learning_algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, base_agent_name="SAC", *args, **kwargs):
        """See `AbstractAgent.default'."""
        test = kwargs.get("test", False)

        from importlib import import_module

        base_agent = getattr(
            import_module("rllib.agent"), f"{base_agent_name}Agent"
        ).default(environment, *args, **kwargs)
        base_algorithm = base_agent.algorithm

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
        reward_model = kwargs.get(
            "reward_model",
            QuadraticReward(
                torch.eye(environment.dim_state[0]),
                torch.eye(environment.dim_action[0]),
                goal=environment.goal,
            ),
        )

        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        model_learning_algorithm = ModelLearningAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_epochs=4 if kwargs.get("test", False) else 30,
            batch_size=64,
            bootstrap=True,
            model_optimizer=model_optimizer,
        )
        simulation_algorithm = SimulationAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            initial_distribution=None,
            max_memory=100000,
            num_subsample=2,
            num_steps=2 if test else 200,
            num_initial_state_samples=8,
            num_initial_distribution_samples=0,
            num_memory_samples=4,
            refresh_interval=2,
        )
        planning_algorithm = PolicyShooting(
            policy=base_agent.policy,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=1,
            gamma=base_agent.gamma,
            num_iter=1,
            num_samples=8,
            num_elites=1,
            action_scale=base_agent.policy.action_scale,
            num_cpu=1,
        )

        return cls(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_learning_algorithm=model_learning_algorithm,
            simulation_algorithm=simulation_algorithm,
            planning_algorithm=planning_algorithm,
            optimizer=base_agent.optimizer,
            num_iter=5 if test else base_agent.num_iter,
            batch_size=base_agent.batch_size,
            policy_opt_gradient_steps=2 if test else base_agent.num_iter,
            dyna_num_samples=15,
            dyna_num_steps=1,
            thompson_sampling=False,
            comment=environment.name,
            *args,
            **kwargs,
        )
