"""Model-Based Soft Actor Critic Agent."""
import torch.nn as nn

from rllib.algorithms.sac import MBSoftActorCritic
from rllib.value_function import NNEnsembleQFunction

from .model_based_agent import ModelBasedAgent


class MBSACAgent(ModelBasedAgent):
    """Implementation of Model-Based SAC Agent."""

    def __init__(self,
                 model_optimizer,
                 policy, q_function, dynamical_model, reward_model,
                 optimizer,
                 initial_distribution=None,
                 plan_horizon=1, plan_samples=8, plan_elites=1,
                 max_memory=10000,
                 model_learn_batch_size=64,
                 model_learn_num_iter=30,
                 bootstrap=True,
                 sac_eta=0.2,
                 sac_epsilon=None,
                 sac_num_iter=100,
                 sac_gradient_steps=50,
                 sac_batch_size=None,
                 sac_action_samples=15,
                 sac_target_num_steps=1,
                 sac_target_update_frequency=4,
                 sac_value_learning_criterion=nn.MSELoss,
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
                 termination=None,
                 tensorboard=False,
                 comment=''):

        q_function = NNEnsembleQFunction.from_q_function(q_function=q_function,
                                                         num_heads=2)
        self.algorithm = MBSoftActorCritic(
            policy, q_function, dynamical_model, reward_model,
            criterion=sac_value_learning_criterion(reduction='mean'),
            eta=sac_eta, epsilon=sac_epsilon, gamma=gamma, termination=termination,
            num_steps=sac_target_num_steps, num_samples=sac_action_samples)
        optimizer = type(optimizer)([p for name, p in self.algorithm.named_parameters()
                                     if ('model' not in name and 'target' not in name)],
                                    **optimizer.defaults)
        self.dist_params = {'tanh': True, 'action_scale': policy.action_scale}

        super().__init__(
            policy=policy, dynamical_model=dynamical_model,
            reward_model=reward_model, model_optimizer=model_optimizer,
            termination=termination, value_function=self.algorithm.value_function,
            plan_horizon=plan_horizon, plan_samples=plan_samples,
            plan_elites=plan_elites,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            bootstrap=bootstrap,
            max_memory=max_memory,
            policy_opt_num_iter=sac_num_iter,
            policy_opt_batch_size=sac_batch_size,
            policy_opt_gradient_steps=sac_gradient_steps,
            policy_opt_target_update_frequency=sac_target_update_frequency,
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
            gamma=gamma, exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard, comment=comment)
