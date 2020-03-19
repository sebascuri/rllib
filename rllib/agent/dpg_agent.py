"""Implementation of Deterministic Policy Gradient Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.dpg import DPG
from rllib.util.logger import Logger
import torch
import numpy as np


class DPGAgent(AbstractAgent):
    """Implementation of the Deterministic Policy Gradient Agent.

    The AbstractDDPGAgent algorithm implements the DPG-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

    TODO: build compatible q-function approximation.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: AbstractPolicy
        policy that is learned.
    exploration: AbstractExplorationStrategy.
        exploration strategy that returns the actions.
    criterion: nn.Module
    critic_optimizer: nn.Optimizer
        q_function optimizer.
    actor_optimizer: nn.Optimizer
        policy optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, q_function, policy, exploration, criterion,
                 critic_optimizer, actor_optimizer, memory, max_action=1,
                 target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1.,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        assert policy.deterministic, "Policy must be deterministic."

        self.dpg_algorithm = DPG(q_function, policy, criterion(reduction='none'), gamma,
                                 policy_noise, noise_clip)

        self.policy = self.dpg_algorithm.policy

        self.exploration = exploration
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.policy_update_frequency = policy_update_frequency

        self.logs['td_errors'] = Logger('abs_mean')
        self.logs['critic_losses'] = Logger('mean')
        self.logs['actor_losses'] = Logger('mean')

    def act(self, state):
        """See `AbstractAgent.act'.

        As the policy is deterministic, some noise must be added to aid exploration.
        """
        action = super().act(state)
        if self._training:
            action += self.exploration()
        return self.max_action * np.clip(action, -1, 1)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch and (self.total_steps > self.exploration_steps) and (
                self.total_episodes > self.exploration_episodes):
            if self._training:
                self._train()
            if self.total_steps % self.target_update_frequency == 0:
                self.dpg_algorithm.update()

    def _train(self):
        """Train the DPG Agent."""
        observation, idx, weight = self.memory.get_batch()
        weight = torch.tensor(weight)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        ans = self.dpg_algorithm(
            observation.state, observation.action, observation.reward,
            observation.next_state, observation.done)

        # Optimize critic
        critic_loss = (weight * ans.critic_loss).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        actor_loss = (weight * ans.actor_loss).mean()
        if not (self.total_steps % self.policy_update_frequency):
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update memory
        self.memory.update(idx, ans.td_error.detach().numpy())

        # Update logs
        self.logs['td_errors'].append(ans.td_error.mean().item())
        self.logs['actor_losses'].append(actor_loss.item())
        self.logs['critic_losses'].append(critic_loss.item())
