"""Implementation of Deterministic Policy Gradient Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset import Observation
from abc import abstractmethod
import torch
import numpy as np
import copy


class AbstractDPGAgent(AbstractAgent):
    """Abstract Implementation of the Deterministic Policy Gradient Algorithm.

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
    """

    def __init__(self, q_function, policy, exploration, criterion, critic_optimizer,
                 actor_optimizer, memory, max_action=1,
                 target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1.,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        assert policy.deterministic, "Policy must be deterministic."

        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)

        self.exploration = exploration
        self.criterion = criterion(reduction='none')
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.policy_update_frequency = policy_update_frequency
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        action += self.exploration()
        return self.max_action * np.clip(action, -1, 1)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch and (self.total_steps > self.exploration_steps) and (
                self.total_episodes > self.exploration_episodes):
            self._train(
                optimize_actor=not (self.total_steps % self.policy_update_frequency))
            if self.total_steps % self.target_update_frequency == 0:
                self.q_target.update_parameters(self.q_function.parameters())
                self.policy_target.update_parameters(self.policy.parameters())

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
            observation, idx, weight = self.memory.get_batch()
            weight = torch.tensor(weight).float()
            observation = Observation(*map(lambda x: x.float(), observation))

            # optimize critic
            td_error = self._train_critic(**observation._asdict(), weight=weight)
            self.memory.update(idx, td_error.detach().numpy())

            # optimize actor
            if optimize_actor:
                self._train_actor(observation.state, weight)

    def _train_critic(self, state, action, reward, next_state, done, weight, *args,
                      **kwargs):
        self.critic_optimizer.zero_grad()
        pred_q, target_q = self._td(state, action, reward, next_state, done)
        if type(pred_q) is not list:
            pred_q = [pred_q]
        loss = torch.zeros_like(target_q)
        td_error = torch.zeros_like(target_q)
        for q in pred_q:
            loss += (weight * self.criterion(q, target_q))
            td_error += q.detach() - target_q.detach()

        loss = loss.mean()
        loss.backward()
        self.critic_optimizer.step()

        td_error_mean = td_error.mean().item()
        self.logs['td_errors'].append(td_error_mean)
        self.logs['episode_td_errors'][-1].append(td_error_mean)
        return td_error

    def _train_actor(self, state, weight):
        self.actor_optimizer.zero_grad()
        action = self.policy(state).mean.clamp(-1, 1)
        q = self.q_function(state.float(), action)
        if type(q) is list:
            q = q[0]
        loss = (-weight * q).mean()
        loss.backward()
        self.actor_optimizer.step()

    @abstractmethod
    def _td(self, state, action, reward, next_state, done, *args, **kwargs):
        raise NotImplementedError
