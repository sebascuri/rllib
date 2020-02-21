"""Implementation of Model-Free Policy Gradient Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from abc import abstractmethod
import copy
from rllib.dataset import Observation
from rllib.dataset.utilities import stack_list_of_tuples


class AbstractPolicyGradient(AbstractAgent):
    """Abstract Implementation of the Actor-Critic Algorithm.

    The AbstractEpisodicPolicyGradient algorithm implements the Actor-Critic algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. NIPS.

    Sutton, R. S., et al. (2000). Policy gradient methods for reinforcement learning
    with function approximation. NIPS.
    """

    eps = 1e-12

    def __init__(self, policy, policy_optimizer, baseline=None, critic=None,
                 baseline_optimizer=None, critic_optimizer=None, criterion=None,
                 num_rollouts=1, target_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.trajectories = []
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.baseline_target = copy.deepcopy(baseline)
        self.baseline_optimizer = baseline_optimizer
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_optimizer = critic_optimizer
        self.criterion = criterion(reduction='none')
        self.num_rollouts = num_rollouts
        self.target_update_freq = target_update_frequency

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == 0:
            self._train()
            self.trajectories = []

        if self.total_episodes % (self.target_update_freq * self.num_rollouts) == 0:
            self.policy_target.parameters = self.policy.parameters

            if self.critic:
                self.critic_target.parameters = self.critic.parameters
            if self.baseline:
                self.baseline_target.parameters = self.baseline.parameters

    def _train(self):
        trajectories = [Observation(*stack_list_of_tuples(t))
                        for t in self.trajectories]

        self._train_actor(trajectories)

        # This could be trained from an off-line dataset ?.
        if self.baseline:
            self._train_baseline(trajectories)
        if self.critic:
            self._train_critic(trajectories)

    def _train_actor(self, trajectories):
        self.policy_optimizer.zero_grad()
        for trajectory in trajectories:
            pi = self.policy(trajectory.state)
            action = trajectory.action
            if self.policy.discrete_action:
                action = trajectory.action.long()
            loss = - pi.log_prob(action) * self._return(*trajectory)
            loss.sum().backward()

        self.policy_optimizer.step()

    def _train_baseline(self, trajectories):
        self.baseline_optimizer.zero_grad()

        for trajectory in zip(trajectories):
            pred_v, target_v = self._td_base(*trajectory)
            loss = self.criterion(pred_v, target_v)
            loss.mean().backward()

        self.baseline_optimizer.step()

    def _train_critic(self, trajectories):
        self.critic_optimizer.zero_grad()

        for trajectory in trajectories:
            pred_q, target_q = self._td_critic(*trajectory)
            loss = self.criterion(pred_q, target_q)
            loss.mean().backward()

        self.critic_optimizer.step()

    def _td_base(self, state, _=None, reward=None, next_state=None, done=None):
        next_v = self.baseline_target(next_state) * (1 - done)
        target_v = reward + self.gamma * next_v
        return self.baseline(state), target_v.detach()

    @abstractmethod
    def _return(self, state, action=None, reward=None, next_state=None, done=None):
        raise NotImplementedError

    @abstractmethod
    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None):
        raise NotImplementedError
