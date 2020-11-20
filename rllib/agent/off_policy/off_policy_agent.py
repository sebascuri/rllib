"""Off Policy Agent."""

import torch

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.experience_replay import ExperienceReplay


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(
        self,
        memory,
        train_frequency=1,
        batch_size=100,
        reset_memory_after_learn=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_frequency=train_frequency, batch_size=batch_size, *args, **kwargs
        )
        self.reset_memory_after_learn = reset_memory_after_learn
        self.memory = memory

    @classmethod
    def default(
        cls, environment, memory=None, max_len=100000, num_steps=0, *args, **kwargs
    ):
        """See `AbstractAgent.default'."""
        if memory is None:
            memory = ExperienceReplay(max_len=100000, num_steps=num_steps)
        return super().default(environment, memory=memory, *args, **kwargs)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)  # this update total steps.
        if self.training:
            self.memory.append(observation)
        if self.train_at_observe and len(self.memory) >= self.batch_size:
            self.learn()

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.train_at_end_episode and len(self.memory) >= self.batch_size:
            self.learn()

        if len(self.memory) > 0 and self.training:  # Maybe learn() resets the memory.
            self.memory.end_episode()

        super().end_episode()  # this update total episodes.

    def learn(self):
        """Train the off-policy agent."""
        #

        def closure():
            """Gradient calculation."""
            observation, idx, weight = self.memory.sample_batch(self.batch_size)

            self.optimizer.zero_grad()
            losses_ = self.algorithm(observation.clone())
            loss = (losses_.combined_loss * weight.detach()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            # Update memory
            self.memory.update(idx, losses_.td_error.abs().detach())

            return losses_

        self._learn_steps(closure)

        if self.reset_memory_after_learn:
            self.memory.reset()
