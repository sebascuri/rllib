"""On Policy Agent."""
import torch

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import stack_list_of_tuples


class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, batch_size=1, num_iter=1, *args, **kwargs):
        super().__init__(num_rollouts=kwargs.pop("num_rollouts", 1), *args, **kwargs)
        self.trajectories = []
        self.batch_size = batch_size
        self.num_iter = num_iter

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)
        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and len(self.trajectories[-1]) >= self.batch_size  # enough data.
            and self.train_frequency > 0  # train after train_frequency transitions.
            and self.total_steps % self.train_frequency == 0  # correct steps.
        ):
            self.learn()
            self.trajectories[-1] = []

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.num_rollouts > 0  # train after num_rollouts transitions.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        ):
            self.learn()
            self.trajectories = list()
        if self.num_rollouts == 0:
            self.trajectories = list()

        super().end_episode()

    def learn(self):
        """Train Policy Gradient Agent."""
        trajectories = [stack_list_of_tuples(t) for t in self.trajectories]

        def closure():
            """Gradient calculation."""
            self.optimizer.zero_grad()
            losses = self.algorithm(trajectories)
            losses.combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            return losses

        self._learn_steps(closure, num_iter=self.num_iter)
