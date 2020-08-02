"""Model-Based DYNA Agent."""
# from itertools import chain

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
        model_optimizer,
        dynamical_model,
        reward_model,
        optimizer,
        termination=None,
        initial_distribution=None,
        plan_horizon=1,
        plan_samples=8,
        plan_elites=1,
        max_memory=10000,
        model_learn_batch_size=64,
        model_learn_num_iter=30,
        bootstrap=True,
        policy_opt_num_iter=100,
        policy_opt_gradient_steps=50,
        policy_opt_batch_size=100,
        dyna_num_samples=15,
        dyna_num_steps=1,
        policy_opt_target_update_frequency=4,
        policy_update_frequency=1,
        sim_num_steps=200,
        sim_initial_states_num_trajectories=8,
        sim_initial_dist_num_trajectories=0,
        sim_memory_num_trajectories=0,
        sim_refresh_interval=0,
        sim_num_subsample=1,
        sim_max_memory=10000,
        thompson_sampling=False,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        self.algorithm = DynaAlgorithm(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination=termination,
            num_steps=dyna_num_steps,
            num_samples=dyna_num_samples,
        )

        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **optimizer.defaults,
        )
        self.dist_params = {
            "tanh": True,
            "action_scale": self.algorithm.policy.action_scale,
        }

        super().__init__(
            policy=self.algorithm.policy,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_optimizer=model_optimizer,
            termination=termination,
            value_function=self.algorithm.value_function,
            plan_horizon=plan_horizon,
            plan_samples=plan_samples,
            plan_elites=plan_elites,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            bootstrap=bootstrap,
            max_memory=max_memory,
            policy_opt_num_iter=policy_opt_num_iter,
            policy_opt_batch_size=policy_opt_batch_size,
            policy_opt_gradient_steps=policy_opt_gradient_steps,
            policy_opt_target_update_frequency=policy_opt_target_update_frequency,
            policy_update_frequency=policy_update_frequency,
            optimizer=optimizer,
            sim_num_steps=sim_num_steps,
            sim_initial_states_num_trajectories=sim_initial_states_num_trajectories,
            sim_initial_dist_num_trajectories=sim_initial_dist_num_trajectories,
            sim_memory_num_trajectories=sim_memory_num_trajectories,
            sim_refresh_interval=sim_refresh_interval,
            sim_num_subsample=sim_num_subsample,
            sim_max_memory=sim_max_memory,
            initial_distribution=initial_distribution,
            thompson_sampling=thompson_sampling,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )

    @classmethod
    def default(
        cls,
        environment,
        base_agent_name="SAC",
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        from importlib import import_module

        base_agent = hasattr(
            import_module("rllib.agent"), f"{base_agent_name}Agent"
        ).default(
            environment,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=False,
            test=test,
        )
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
        dynamical_model = TransformedModel(model, list())
        reward_model = QuadraticReward(
            torch.eye(environment.dim_state[0]),
            torch.eye(environment.dim_action[0]),
            goal=environment.goal,
        )

        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        return cls(
            base_algorithm=base_algorithm,
            model_optimizer=model_optimizer,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            optimizer=base_agent.optimizer,
            termination=None,
            initial_distribution=None,
            plan_horizon=1,
            plan_samples=8,
            plan_elites=1,
            max_memory=10000,
            model_learn_batch_size=64,
            model_learn_num_iter=4 if test else 30,
            bootstrap=True,
            policy_opt_num_iter=5 if test else 100,
            policy_opt_gradient_steps=5 if test else base_algorithm.num_iter,
            policy_opt_batch_size=base_algorithm.batch_size,
            policy_update_frequency=base_algorithm.policy_update_frequency,
            policy_opt_target_update_frequency=base_algorithm.target_update_frequency,
            dyna_num_samples=15,
            dyna_num_steps=1,
            sim_num_steps=5 if test else 200,
            sim_initial_states_num_trajectories=8,
            sim_initial_dist_num_trajectories=0,
            sim_memory_num_trajectories=0,
            sim_refresh_interval=0,
            sim_num_subsample=1,
            sim_max_memory=10000,
            thompson_sampling=False,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
