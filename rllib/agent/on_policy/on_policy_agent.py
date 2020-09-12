"""On Policy Agent."""
import torch

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import stack_list_of_tuples


class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, num_rollouts=1, *args, **kwargs):
        super().__init__(num_rollouts=num_rollouts, *args, **kwargs)

        self.trajectories = []

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)
        if self.train_at_observe and len(self.trajectories[-1]) >= self.batch_size:
            self.learn()
            self.trajectories[-1] = []

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.train_at_end_episode:
            self.learn()
            self.trajectories = list()
        if self.num_rollouts == 0:
            self.trajectories = list()

        super().end_episode()

    def learn(self):
        """Train Policy Gradient Agent."""
        trajectories = [stack_list_of_tuples(t).clone() for t in self.trajectories]

        def closure():
            """Gradient calculation."""
            self.optimizer.zero_grad()
            losses = self.algorithm(trajectories)
            losses.combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            return losses

        self._learn_steps(closure)
