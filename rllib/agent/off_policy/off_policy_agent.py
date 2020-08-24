"""Off Policy Agent."""

import torch

from rllib.agent.abstract_agent import AbstractAgent


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(
        self,
        memory,
        num_iter=1,
        batch_size=64,
        target_update_frequency=1,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_frequency=kwargs.pop("train_frequency", 1), *args, **kwargs
        )

        self.batch_size = batch_size
        self.memory = memory

        self.target_update_frequency = target_update_frequency
        self.num_iter = num_iter

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)  # this update total steps.
        self.memory.append(observation)
        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and len(self.memory) >= self.batch_size  # enough data.
            and self.train_frequency > 0  # train after a transition.
            and self.total_steps % self.train_frequency == 0  # correct steps.
        ):
            self.learn()

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and len(self.memory) > self.batch_size  # enough data.
            and self.num_rollouts > 0  # train once the episode ends.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        ):  # use total_episodes + 1 because the super() is called after training.
            self.learn()

        if len(self.memory) > 0:  # Maybe learn() resets the memory.
            self.memory.end_episode()

        super().end_episode()  # this update total episodes.

    def learn(self):
        """Train the off-policy agent."""
        #

        def closure():
            """Gradient calculation."""
            observation, idx, weight = self.memory.sample_batch(self.batch_size)

            self.optimizer.zero_grad()
            losses_ = self.algorithm(observation)
            loss = (losses_.combined_loss.squeeze(-1) * weight.detach()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            # Update memory
            self.memory.update(idx, losses_.td_error.abs().detach())

            return losses_

        self._learn_steps(closure, num_iter=self.num_iter)
