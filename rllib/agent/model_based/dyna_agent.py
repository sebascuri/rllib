"""Model-Based DYNA Agent."""

import torch
from torch.optim import Adam

from rllib.algorithms.dyna_algorithm import DynaAlgorithm
from rllib.model import EnsembleModel, TransformedModel
from rllib.reward.quadratic_reward import QuadraticReward

from .model_based_agent import ModelBasedAgent


class DynaAgent(ModelBasedAgent):
    """Implementation of a Dyna-Agent.

    A Dyna Agent gets a model-free algorithm and uses the model to simulate transitions.
    """

    def __init__(
        self,
        base_algorithm,
        dynamical_model,
        reward_model,
        model_optimizer,
        optimizer,
        dyna_num_steps,
        dyna_num_samples,
        termination=None,
        policy=None,
        *args,
        **kwargs,
    ):
        self.algorithm = DynaAlgorithm(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination=termination,
            num_steps=dyna_num_steps,
            num_samples=dyna_num_samples,
        )

        if hasattr(self.algorithm, "policy"):
            policy = self.algorithm.policy

        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **optimizer.defaults,
        )
        self.dist_params = {"tanh": True, "action_scale": policy.action_scale}

        super().__init__(
            policy=policy,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_optimizer=model_optimizer,
            termination=termination,
            value_function=self.algorithm.value_function,
            optimizer=optimizer,
            *args,
            **kwargs,
        )

    @classmethod
    def default(cls, environment, base_agent_name="SAC", *args, **kwargs):
        """See `AbstractAgent.default'."""
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

        test = kwargs.get("test", False)
        return cls(
            base_algorithm=base_algorithm,
            model_optimizer=model_optimizer,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            optimizer=base_agent.optimizer,
            termination=None,
            initial_distribution=None,
            plan_horizon=0 if test else 5,
            plan_samples=8,
            plan_elites=1,
            max_memory=10000,
            model_learn_batch_size=64,
            model_learn_num_iter=4 if test else 30,
            bootstrap=True,
            policy_opt_num_iter=5 if test else 100,
            policy_opt_gradient_steps=2 if test else base_agent.num_iter,
            policy_opt_batch_size=100,
            policy_update_frequency=1,
            policy_opt_target_update_frequency=1,
            dyna_num_samples=15,
            dyna_num_steps=1,
            sim_num_steps=2 if test else 200,
            sim_initial_states_num_trajectories=8,
            sim_initial_dist_num_trajectories=0,
            sim_memory_num_trajectories=0,
            sim_refresh_interval=0,
            sim_num_subsample=1,
            sim_max_memory=10000,
            thompson_sampling=False,
            comment=environment.name,
            *args,
            **kwargs,
        )
