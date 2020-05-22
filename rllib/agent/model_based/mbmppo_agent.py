"""Model-Based MPPO Agent."""

from .model_based_agent import ModelBasedAgent


class MBMPPOAgent(ModelBasedAgent):
    """Implementation of Model-Based MPPO Agent."""

    def __init__(self,
                 env_name,
                 mppo,
                 model_optimizer,
                 mppo_optimizer,
                 initial_distribution=None,
                 plan_horizon=1, plan_samples=8, plan_elite=1,
                 max_memory=10000,
                 model_learn_batch_size=64,
                 model_learn_num_iter=30,
                 mppo_num_iter=100,
                 mppo_gradient_steps=50,
                 mppo_batch_size=None,
                 mppo_target_update_frequency=4,
                 sim_num_steps=200,
                 sim_initial_states_num_trajectories=8,
                 sim_initial_dist_num_trajectories=0,
                 sim_memory_num_trajectories=0,
                 sim_num_subsample=1,
                 thompson_sampling=False,
                 gamma=1.0,
                 exploration_steps=0,
                 exploration_episodes=0,
                 comment=''):
        super().__init__(
            env_name, policy=mppo.policy, dynamical_model=mppo.dynamical_model,
            reward_model=mppo.reward_model, model_optimizer=model_optimizer,
            termination=mppo.termination, value_function=mppo.value_function,
            plan_horizon=plan_horizon, plan_samples=plan_samples, plan_elite=plan_elite,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            max_memory=max_memory,
            policy_opt_num_iter=mppo_num_iter,
            policy_opt_batch_size=mppo_batch_size,
            policy_opt_gradient_steps=mppo_gradient_steps,
            policy_opt_target_update_frequency=mppo_target_update_frequency,
            optimizer=mppo_optimizer,
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

        self.algorithm = mppo
        self.value_function = self.mppo.value_function

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
