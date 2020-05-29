"""MPC Agent Implementation."""

import numpy as np
import torch
import torch.nn as nn

from rllib.algorithms.td import ModelBasedTDLearning
from rllib.policy.mpc_policy import MPCPolicy

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(self, env_name, mpc_policy: MPCPolicy,
                 model_learn_num_iter=0,
                 model_learn_batch_size=64,
                 model_optimizer=None,
                 value_optimizer=None,
                 max_memory=1,
                 value_opt_num_iter=0,
                 value_opt_batch_size=None,
                 value_num_steps_returns=1,
                 value_gradient_steps=50,
                 sim_num_steps=0,
                 sim_initial_states_num_trajectories=0,
                 sim_initial_dist_num_trajectories=0,
                 sim_memory_num_trajectories=0,
                 initial_distribution=None,
                 thompson_sampling=False,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(
            env_name,
            dynamical_model=mpc_policy.solver.dynamical_model,
            reward_model=mpc_policy.solver.reward_model,
            policy=mpc_policy,
            value_function=mpc_policy.solver.terminal_reward,
            termination=mpc_policy.solver.termination,
            model_optimizer=model_optimizer,
            plan_horizon=0,  # Calling the mpc policy already plans.
            plan_samples=0,
            plan_elites=0,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            policy_opt_num_iter=value_opt_num_iter,
            policy_opt_batch_size=value_opt_batch_size,
            max_memory=max_memory,
            sim_num_steps=sim_num_steps,
            sim_initial_states_num_trajectories=sim_initial_states_num_trajectories,
            sim_initial_dist_num_trajectories=sim_initial_dist_num_trajectories,
            sim_memory_num_trajectories=sim_memory_num_trajectories,
            initial_distribution=initial_distribution,
            thompson_sampling=thompson_sampling,
            gamma=gamma, exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes, comment=comment)

        self.value_optimizer = value_optimizer
        self.value_gradient_steps = value_gradient_steps
        self.value_learning = ModelBasedTDLearning(
            self.value_function, criterion=nn.MSELoss(reduction='none'),
            policy=self.plan_policy, dynamical_model=self.dynamical_model,
            reward_model=self.reward_model, termination=self.termination,
            num_steps=value_num_steps_returns, gamma=self.gamma)

    def _optimize_policy(self):
        """Optimize policy by optimizing value function."""
        # Iterate over state batches in the state distribution
        states = self.sim_trajectory.state.reshape(-1, self.dynamical_model.dim_state)
        np.random.shuffle(states.numpy())
        state_batches = torch.split(
            states, self.policy_opt_batch_size)[::self.sim_num_subsample]

        for _ in range(self.value_gradient_steps):
            # obs, _, _ = self.sim_dataset.get_batch(self.policy_opt_batch_size)
            # states = obs.state[:, 0]
            idx = np.random.choice(len(state_batches))
            states = state_batches[idx]

            self.plan_policy.reset()
            self.value_learning.zero_grad()
            losses = self.value_learning(states.unsqueeze(-2))
            loss = losses.loss.mean()
            loss.backward()
            self.value_optimizer.step()

            self.logger.update(critic_losses=loss.item(),
                               td_errors=losses.td_error.abs().mean().item())
