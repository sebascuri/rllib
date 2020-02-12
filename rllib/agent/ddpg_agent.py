"""Implementation of DDPG Algorithms."""
from .abstract_agent import AbstractAgent
import torch
import numpy as np
import copy


class DDPGAgent(AbstractAgent):
    """Abstract Implementation of the DDPG Algorithm.

    The AbstractDDPGAgent algorithm implements the DDPT-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

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
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, q_function, policy, exploration, criterion, critic_optimizer,
                 actor_optimizer, memory, max_action=1,
                 target_update_frequency=4, policy_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)

        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)

        self.exploration = exploration
        self.criterion = criterion
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.policy_update_frequency = policy_update_frequency

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        action += self.exploration()
        return np.clip(action, -self.max_action, self.max_action)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch and (self.total_steps > self.exploration_steps) and (
                self.total_episodes > self.exploration_episodes):
            self._train(
                optimize_actor=not (self.total_steps % self.policy_update_frequency))
            if self.total_steps % self.target_update_frequency == 0:
                self.q_target.parameters = self.q_function.parameters
                self.policy_target.parameters = self.policy.parameters

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    def _train(self, batches=1, optimize_actor=True):
        """Train the DDPG for `batches' batches.

        Parameters
        ----------
        batches: int

        """
        for batch in range(batches):
            observation, idx, w = self.memory.get_batch()
            w = torch.tensor(w).float()
            state = observation.state.float()
            action = observation.action.float()
            reward = observation.reward.float()
            next_state = observation.next_state.float()
            done = observation.done.float()

            # optimize critic
            td_error = self._train_critic(state, action, reward, next_state, done, w)
            self.memory.update(idx, td_error.detach().numpy())

            # optimize actor
            if optimize_actor:
                self._train_actor(state, w)

    def _train_critic(self, state, action, reward, next_state, done, weight):
        self.critic_optimizer.zero_grad()
        pred_q, target_q = self._td(state.float(), action.float(), reward.float(),
                                    next_state.float(), done.float())
        loss = self.criterion(pred_q, target_q, reduction='none')
        loss = weight * loss
        loss.mean().backward()
        self.critic_optimizer.step()

        td_error = pred_q.detach() - target_q.detach()
        td_error_mean = td_error.mean().item()
        self.logs['td_errors'].append(td_error_mean)
        self.logs['episode_td_errors'][-1].append(td_error_mean)
        return td_error

    def _train_actor(self, state, weight):
        self.actor_optimizer.zero_grad()
        policy_action = self.policy(state).loc
        q = -self.q_function(state.float(), policy_action)
        loss = weight * q
        loss.mean().backward()
        self.actor_optimizer.step()

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * Q_target(x', \pi_target(x'))
        next_policy_action = self.policy_target(next_state).loc
        next_q = self.q_target(next_state, next_policy_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
