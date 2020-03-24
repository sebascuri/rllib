"""Model-Based MPPO Agent."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_agent import AbstractAgent
from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import disable_gradient
from rllib.util.rollout import rollout_model
from rllib.util.training import train_model


class MBMPPOAgent(AbstractAgent):
    """Implementation of Model-Based MPPO Agent.

    Parameters
    ----------
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    """

    def __init__(self, dynamic_model, mppo,
                 model_optimizer, mppo_optimizer, transformations,
                 max_len=10000, batch_size=64,
                 num_env_rollouts=1,
                 num_iter=30,
                 num_mppo_iter=100,
                 num_simulation_steps=200,
                 num_simulation_trajectories=8,
                 state_refresh_interval=2,
                 termination=None,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)

        self.mppo = mppo
        self.mppo_optimizer = mppo_optimizer

        self.dynamic_model = dynamic_model
        self.model_optimizer = model_optimizer

        self.dataset = ExperienceReplay(max_len=max_len, batch_size=batch_size,
                                        transformations=transformations)

        self.policy = self.mppo.policy

        # Model Learning parameters.
        self.num_iter = num_iter

        # MPPO specific parameters.
        self.num_mppo_iter = num_mppo_iter
        self.batch_size = batch_size
        self.num_env_rollouts = num_env_rollouts
        self.num_simulation_steps = num_simulation_steps
        self.num_simulation_trajectories = num_simulation_trajectories
        self.state_refresh_interval = state_refresh_interval

        self.termination = termination

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.dataset.append(observation)

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_env_rollouts == 0:
            if self._training:
                self._train()

        super().end_episode()
        print(f"Episode Returns {self.logs['rewards'].episode_log[-1]}")

    def _train(self) -> None:
        # Step 1: Train Model with new data.
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        model_train_log = train_model(
            self.dynamic_model.base_model, train_loader=loader,
            max_iter=self.num_iter, optimizer=self.model_optimizer, print_flag=False)
        print(f"Model training loss: {model_train_log.episode_log[-1]}")

        # Step 2: Optimize Policy with model.
        init_distribution = torch.distributions.Uniform(torch.tensor([-np.pi, -0.05]),
                                                        torch.tensor([np.pi, 0.05]))

        # self.mppo.dynamic_model = self.dynamic_model
        if self.total_steps <= self.exploration_steps or (
                self.total_episodes <= self.exploration_episodes):
            return
        with disable_gradient(self.dynamic_model):
            policy_returns = []
            for i in tqdm(range(self.num_mppo_iter)):
                # Compute the state distribution
                if i % self.state_refresh_interval == 0:
                    with torch.no_grad():
                        initial_states = init_distribution.sample(
                            (self.num_simulation_trajectories,))
                        trajectory = rollout_model(self.mppo.dynamical_model,
                                                   reward_model=self.mppo.reward_model,
                                                   policy=self.mppo.policy,
                                                   initial_state=initial_states,
                                                   max_steps=self.num_simulation_steps,
                                                   termination=self.termination)
                        stacked_trajectory = Observation(
                            *stack_list_of_tuples(trajectory))
                        policy_returns.append(
                            stacked_trajectory.reward.sum(dim=0).mean().item())
                        # Shuffle to get a state distribution
                        states = stacked_trajectory.state.reshape(-1,
                                                                  self.policy.dim_state)
                        np.random.shuffle(states.numpy())
                        state_batches = torch.split(states, self.batch_size)

                # Copy over old policy for KL divergence
                self.mppo.reset()

                # Iterate over state batches in the state distribution
                for states in state_batches:
                    self.mppo_optimizer.zero_grad()
                    losses = self.mppo(states)
                    losses.loss.backward()
                    self.mppo_optimizer.step()

        print(f"Policy Returns {np.sum(policy_returns)}")
        print(f"Losses {losses.loss.item()}")
