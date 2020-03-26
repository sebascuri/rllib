"""Implementation of REINFORCE Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.reinforce import REINFORCE
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples


class REINFORCEAgent(AbstractAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, environment, policy, policy_optimizer, baseline=None,
                 baseline_optimizer=None, criterion=None, num_rollouts=1,
                 target_update_frequency=1, gamma=1.0, exploration_steps=0,
                 exploration_episodes=0):
        super().__init__(environment, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.trajectories = []
        self.reinforce = REINFORCE(policy, baseline,
                                   criterion(reduction='none'),
                                   gamma)
        self.policy = self.reinforce.policy
        self.policy_optimizer = policy_optimizer
        self.baseline_optimizer = baseline_optimizer

        self.num_rollouts = num_rollouts
        self.target_update_frequency = target_update_frequency

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)
        if self.total_steps % self.target_update_frequency == 0:
            self.reinforce.update()

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == 0:
            if self._training:
                self._train()
            self.trajectories = list()

        super().end_episode()

    def _train(self):
        """See `AbstractAgent.train_agent'."""
        trajectories = [Observation(*stack_list_of_tuples(t))
                        for t in self.trajectories]

        # Update actor.
        self.policy_optimizer.zero_grad()
        if self.baseline_optimizer is not None:
            self.baseline_optimizer.zero_grad()

        losses = self.reinforce(trajectories)
        losses.actor_loss.backward()
        self.policy_optimizer.step()

        # Update Logs.
        self.logger.update(actor_losses=losses.actor_loss.item())

        # Update baseline.
        if self.baseline_optimizer is not None:
            losses.baseline_loss.backward()
            self.baseline_optimizer.step()
            self.logger.update(baseline_losses=losses.baseline_loss.item())
