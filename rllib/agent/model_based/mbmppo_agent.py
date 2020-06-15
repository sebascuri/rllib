"""Model-Based MPPO Agent."""
from rllib.algorithms.mppo import MBMPPO

from .model_based_agent import ModelBasedAgent


class MBMPPOAgent(ModelBasedAgent):
    """Implementation of Model-Based MPPO Agent."""

    def __init__(
        self,
        model_optimizer,
        policy,
        value_function,
        dynamical_model,
        reward_model,
        optimizer,
        mppo_value_learning_criterion,
        termination=None,
        initial_distribution=None,
        plan_horizon=1,
        plan_samples=8,
        plan_elites=1,
        max_memory=10000,
        model_learn_batch_size=64,
        model_learn_num_iter=30,
        bootstrap=True,
        mppo_epsilon=None,
        mppo_epsilon_mean=None,
        mppo_epsilon_var=None,
        mppo_eta=None,
        mppo_eta_mean=None,
        mppo_eta_var=None,
        mppo_num_iter=100,
        mppo_gradient_steps=50,
        mppo_batch_size=None,
        mppo_num_action_samples=15,
        mppo_target_update_frequency=4,
        sim_num_steps=200,
        sim_initial_states_num_trajectories=8,
        sim_initial_dist_num_trajectories=0,
        sim_memory_num_trajectories=0,
        sim_max_memory=100000,
        sim_num_subsample=1,
        sim_refresh_interval=1,
        thompson_sampling=False,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):

        self.algorithm = MBMPPO(
            dynamical_model,
            reward_model,
            policy,
            value_function,
            criterion=mppo_value_learning_criterion,
            epsilon=mppo_epsilon,
            epsilon_mean=mppo_epsilon_mean,
            epsilon_var=mppo_epsilon_var,
            eta=mppo_eta,
            eta_mean=mppo_eta_mean,
            eta_var=mppo_eta_var,
            num_action_samples=mppo_num_action_samples,
            gamma=gamma,
            termination=termination,
        )
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name)
            ],
            **optimizer.defaults,
        )

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
            policy_opt_num_iter=mppo_num_iter,
            policy_opt_batch_size=mppo_batch_size,
            policy_opt_gradient_steps=mppo_gradient_steps,
            policy_opt_target_update_frequency=mppo_target_update_frequency,
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
