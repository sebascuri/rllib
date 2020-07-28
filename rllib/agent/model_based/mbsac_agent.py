"""Model-Based Soft Actor Critic Agent."""
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sac import MBSoftActorCritic
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.value_function import NNEnsembleQFunction, NNQFunction

from .model_based_agent import ModelBasedAgent


class MBSACAgent(ModelBasedAgent):
    """Implementation of Model-Based SAC Agent."""

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
        sac_value_learning_criterion=nn.MSELoss,
        sac_eta=0.2,
        sac_regularization=False,
        sac_num_iter=100,
        sac_gradient_steps=50,
        sac_batch_size=None,
        sac_action_samples=15,
        sac_target_num_steps=1,
        sac_target_update_frequency=4,
        sac_policy_update_frequency=1,
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
        q_function = NNEnsembleQFunction.from_q_function(
            q_function=q_function, num_heads=2
        )
        self.algorithm = MBSoftActorCritic(
            policy,
            q_function,
            dynamical_model,
            reward_model,
            criterion=sac_value_learning_criterion(reduction="mean"),
            eta=sac_eta,
            regularization=sac_regularization,
            gamma=gamma,
            termination=termination,
            num_steps=sac_target_num_steps,
            num_samples=sac_action_samples,
        )
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name)
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
            plan_horizon=plan_horizon,
            plan_samples=plan_samples,
            plan_elites=plan_elites,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            bootstrap=bootstrap,
            max_memory=max_memory,
            policy_opt_num_iter=sac_num_iter,
            policy_opt_batch_size=sac_batch_size,
            policy_opt_gradient_steps=sac_gradient_steps,
            policy_opt_target_update_frequency=sac_target_update_frequency,
            policy_update_frequency=sac_policy_update_frequency,
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
            torch.eye(environment.dim_state),
            torch.eye(environment.dim_action),
            goal=environment.goal,
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
            goal=environment.goal,
            deterministic=False,
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
            sac_value_learning_criterion=loss.MSELoss,
            sac_eta=0.2,
            sac_regularization=False,
            sac_num_iter=5 if test else 100,
            sac_gradient_steps=5 if test else 50,
            sac_batch_size=None,
            sac_action_samples=15,
            sac_target_num_steps=1,
            sac_target_update_frequency=4,
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
