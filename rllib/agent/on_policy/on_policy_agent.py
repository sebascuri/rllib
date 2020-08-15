"""On Policy Agent."""
from dataclasses import asdict

import torch

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import average_dataclass, stack_list_of_tuples


class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(
        self,
        optimizer,
        batch_size=1,
        target_update_frequency=1,
        num_iter=1,
        *args,
        **kwargs,
    ):
        super().__init__(num_rollouts=kwargs.pop("num_rollouts", 1), *args, **kwargs)
        self.trajectories = []
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
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
        for _ in range(self.num_iter):

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses_ = self.algorithm(trajectories)
                losses_.loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.algorithm.parameters(), self.clip_gradient_val
                )

                return losses_

            losses = self.optimizer.step(closure=closure)
            # Update logs
            self.logger.update(**asdict(average_dataclass(losses)))
            self.logger.update(**self.algorithm.info())

            self.counters["train_steps"] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

            if self.early_stop(losses, **self.algorithm.info()):
                break

        self.algorithm.reset()
