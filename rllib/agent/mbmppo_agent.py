"""Model-Based MPPO Agent."""

import numpy as np
import gpytorch
from gym.utils import colorize
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_agent import AbstractAgent
from rllib.dataset.experience_replay import BootstrapExperienceReplay, ExperienceReplay
# from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.policy.derived_policy import DerivedPolicy

from rllib.model import ExactGPModel
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

    def __init__(self, environment, mppo,
                 model_optimizer, mppo_optimizer, transformations,
                 max_memory=10000, batch_size=64,
                 num_model_iter=30,
                 num_mppo_iter=100,
                 num_simulation_steps=200,
                 num_simulation_trajectories=8,
                 state_refresh_interval=2,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(environment, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.mppo = mppo
        self.mppo_optimizer = mppo_optimizer
        self.model_optimizer = model_optimizer
        if hasattr(mppo.dynamical_model.base_model, 'num_heads'):
            num_heads = mppo.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        self.dataset = BootstrapExperienceReplay(
            max_len=max_memory, num_bootstraps=num_heads,
            transformations=transformations)
        self.sim_dataset = ExperienceReplay(max_len=max_memory)

        if self.mppo.policy.dim_action == mppo.dynamical_model.dim_action:
            self.policy = self.mppo.policy
        else:
            self.policy = DerivedPolicy(self.mppo.policy)

        # Model Learning parameters.
        self.num_model_iter = num_model_iter

        # MPPO specific parameters.
        self.num_mppo_iter = num_mppo_iter
        self.batch_size = batch_size
        self.num_simulation_steps = num_simulation_steps
        self.num_simulation_trajectories = num_simulation_trajectories
        self.state_refresh_interval = state_refresh_interval

        self.initial_states = torch.tensor(float('nan'))
        self.new_episode = True
        self.trajectory = []
        self.sim_trajectory = []

        layout = {
            'Model Training': {
                'current': ['Multiline',
                            [f"current/model-{i}" for i in range(num_heads)] + [
                                "current/model_loss"]],
                'episode': ['Multiline',
                            [f"episode/model-{i}" for i in range(num_heads)] + [
                                "episode/model_loss"]],
            },
            'Policy Training': {
                'current': ['Multiline', ["current/value_loss", "current/policy_loss",
                                          "current/eta_loss"]],
                'episode': ['Multiline', ["episode/value_loss", "episode/policy_loss",
                                          "episode/eta_loss"]],
            },
            'Returns': {
                'current': ['Multiline', ["current/rewards",
                                          "current/model_return"]],
                'episode': ['Multiline', ["episode/environment_return",
                                          "episode/model_return"]]
            }
        }
        self.logger.writer.add_custom_scalars(layout)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.dataset.append(observation)
        self.trajectory.append(observation)
        if self.new_episode:
            initial_state = observation.state.unsqueeze(0)
            if torch.isnan(self.initial_states).any():
                self.initial_states = initial_state
            else:
                self.initial_states = torch.cat((self.initial_states, initial_state))
            self.new_episode = False

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectory = []
        self.new_episode = True

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self._training:
            if isinstance(self.mppo.dynamical_model.base_model, ExactGPModel):
                print(colorize('Update GP Model', 'yellow'))
                observation = stack_list_of_tuples(self.trajectory)
                for transform in self.dataset.transformations:
                    observation = transform(observation)

                self.mppo.dynamical_model.base_model.add_data(
                    observation.state, observation.action, observation.next_state)

            self._train()
        super().end_episode()

    def _train(self) -> None:
        # Step 1: Train Model with new data.
        print(colorize('Training Model', 'yellow'))
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        train_model(self.mppo.dynamical_model.base_model, train_loader=loader,
                    max_iter=self.num_model_iter, optimizer=self.model_optimizer,
                    logger=self.logger)

        if hasattr(self.mppo.dynamical_model.base_model, 'num_heads'):
            self.mppo.dynamical_model.base_model.select_head(
                self.mppo.dynamical_model.base_model.num_heads)

        # Step 2: Optimize Policy with model.
        if self.total_steps < self.exploration_steps or (
                self.total_episodes < self.exploration_episodes):
            return

        print(colorize('Optimizing Policy with Model Data', 'yellow'))
        self.mppo.dynamical_model.eval()
        with disable_gradient(self.mppo.dynamical_model), \
                gpytorch.settings.fast_pred_var():
            for i in tqdm(range(self.num_mppo_iter)):
                # Compute the state distribution
                if i % self.state_refresh_interval == 0:
                    with torch.no_grad():
                        idx = torch.randint(self.initial_states.shape[0],
                                            (1, self.num_simulation_trajectories,))
                        initial_states = self.initial_states[idx]
                        trajectory = rollout_model(self.mppo.dynamical_model,
                                                   reward_model=self.mppo.reward_model,
                                                   policy=self.mppo.policy,
                                                   initial_state=initial_states,
                                                   max_steps=self.num_simulation_steps,
                                                   termination=self.mppo.termination)
                        self.sim_trajectory = stack_list_of_tuples(trajectory)

                        # Sum along trajectory, average across samples
                        average_return = self.sim_trajectory.reward.sum(dim=0).mean()
                        self.logger.update(model_return=average_return.item())

                        self.logger.update(total_scale=(
                            torch.diagonal(self.sim_trajectory.next_state_scale_tril,
                                           dim1=-1, dim2=-2)
                        ).sum(dim=0).mean())

                        # Shuffle to get a state distribution
                        states = self.sim_trajectory.state.reshape(
                            -1, self.mppo.dynamical_model.dim_state)
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

                    self.logger.update(**losses._asdict())
