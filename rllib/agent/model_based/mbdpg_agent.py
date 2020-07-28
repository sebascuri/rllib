"""Model-Based Deterministic Policy Gradient Agent."""
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam

from rllib.algorithms.dpg import MBDPG
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.value_function import NNEnsembleQFunction, NNQFunction

from .model_based_agent import ModelBasedAgent


class MBDPGAgent(ModelBasedAgent):
    """Implementation of Model-Based DPG Agent."""

    def __init__(
        self,
        model_optimizer,
        policy,
        q_function,
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
        dpg_value_learning_criterion=nn.MSELoss,
        dpg_num_iter=100,
        dpg_gradient_steps=50,
        dpg_batch_size=None,
        dpg_action_samples=15,
        dpg_target_num_steps=1,
        dpg_target_update_frequency=4,
        dpg_noise_clip=1.0,
        dpg_policy_noise=1.0,
        dpg_as_td3=False,
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
        if dpg_as_td3:
            q_function = NNEnsembleQFunction.from_q_function(
                q_function=q_function, num_heads=2
            )
        self.algorithm = MBDPG(
            policy,
            q_function,
            dynamical_model,
            reward_model,
            noise_clip=dpg_noise_clip,
            policy_noise=dpg_policy_noise,
            criterion=dpg_value_learning_criterion(reduction="mean"),
            gamma=gamma,
            termination=termination,
            num_steps=dpg_target_num_steps,
            num_samples=dpg_action_samples,
        )
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name)
            ],
            **optimizer.defaults,
        )
        self.dist_params = {"action_scale": policy.action_scale}

        super().__init__(
            policy=policy,
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
            policy_opt_num_iter=dpg_num_iter,
            policy_opt_batch_size=dpg_batch_size,
            policy_opt_gradient_steps=dpg_gradient_steps,
            policy_opt_target_update_frequency=dpg_target_update_frequency,
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
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
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
        dynamical_model = TransformedModel(model, list())
        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        reward_model = QuadraticReward(
            torch.eye(environment.dim_state), torch.eye(environment.dim_action)
        )

        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            layers=[100, 100],
            biased_head=True,
            non_linearity="ReLU",
            squashed_output=True,
            input_transform=None,
            action_scale=environment.action_scale,
            deterministic=True,
            tau=5e-3,
        )

        q_function = NNQFunction(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            layers=[200, 200],
            biased_head=True,
            non_linearity="ReLU",
            input_transform=None,
            tau=5e-3,
        )

        optimizer = Adam(chain(policy.parameters(), q_function.parameters()), lr=5e-3)

        return cls(
            model_optimizer,
            policy,
            q_function,
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
            model_learn_num_iter=4 if test else 30,
            bootstrap=True,
            dpg_num_iter=1,
            dpg_gradient_steps=1,
            dpg_batch_size=None,
            dpg_action_samples=15,
            dpg_target_num_steps=1,
            dpg_target_update_frequency=1,
            dpg_noise_clip=0.5,
            dpg_policy_noise=0.2,
            dpg_as_td3=True,
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
