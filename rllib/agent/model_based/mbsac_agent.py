"""Model-Based Soft Actor Critic Agent."""

import torch.nn as nn
from .model_based_agent import ModelBasedAgent
from rllib.algorithms.sac import MBSoftActorCritic


class MBSACAgent(ModelBasedAgent):
    """Implementation of Model-Based SAC Agent."""

    def __init__(self,
                 env_name,
                 model_optimizer,
                 policy, q_function, dynamical_model, reward_model,
                 actor_optimizer,
                 critic_optimizer,
                 temperature=0.2,
                 criterion=nn.MSELoss,
                 initial_distribution=None,
                 plan_horizon=1, plan_samples=8, plan_elite=1,
                 max_memory=10000,
                 model_learn_batch_size=64,
                 model_learn_num_iter=30,
                 sac_num_iter=100,
                 sac_gradient_steps=50,
                 sac_batch_size=None,
                 sac_action_samples=15,
                 sac_target_num_steps=1,
                 sac_target_update_frequency=4,
                 sim_num_steps=200,
                 sim_initial_states_num_trajectories=8,
                 sim_initial_dist_num_trajectories=0,
                 sim_memory_num_trajectories=0,
                 sim_num_subsample=1,
                 thompson_sampling=False,
                 gamma=1.0,
                 exploration_steps=0,
                 exploration_episodes=0,
                 termination=None,
                 comment=''):
        super().__init__(
            env_name, policy=policy, dynamical_model=dynamical_model,
            reward_model=reward_model, model_optimizer=model_optimizer,
            termination=termination, value_function=None,
            plan_horizon=plan_horizon, plan_samples=plan_samples, plan_elite=plan_elite,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            max_memory=max_memory,
            policy_opt_num_iter=sac_num_iter,
            policy_opt_batch_size=sac_batch_size,
            policy_opt_gradient_steps=sac_gradient_steps,
            policy_opt_target_update_frequency=sac_target_update_frequency,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            sim_num_steps=sim_num_steps,
            sim_initial_states_num_trajectories=sim_initial_states_num_trajectories,
            sim_initial_dist_num_trajectories=sim_initial_dist_num_trajectories,
            sim_memory_num_trajectories=sim_memory_num_trajectories,
            sim_refresh_interval=1,
            sim_num_subsample=sim_num_subsample,
            initial_distribution=initial_distribution,
            thompson_sampling=thompson_sampling,
            gamma=gamma, exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes, comment=comment)

        self.algorithm = MBSoftActorCritic(
            policy, q_function, dynamical_model, reward_model,
            criterion=criterion(reduction='mean'), temperature=temperature,
            gamma=gamma, termination=termination,
            num_steps=sac_target_num_steps, num_samples=sac_action_samples)
        self.value_function = None

        if hasattr(self.dynamical_model.base_model, 'num_heads'):
            num_heads = self.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        layout = {
            'Model Training': {
                'average': ['Multiline',
                            [f"average/model-{i}" for i in range(num_heads)] + [
                                "average/model_loss"]],
            },
            'Policy Training': {
                'average': ['Multiline', ["average/value_loss", "average/policy_loss",
                                          "average/eta_loss"]],
            },
            'Returns': {
                'average': ['Multiline', ["average/environment_return",
                                          "average/model_return"]]
            }
        }
        self.logger.writer.add_custom_scalars(layout)
