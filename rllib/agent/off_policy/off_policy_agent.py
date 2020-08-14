"""Off Policy Agent."""
import contextlib

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import average_named_tuple
from rllib.util.neural_networks.utilities import DisableGradient


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(
        self,
        memory,
        optimizer,
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

        self.optimizer = optimizer
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
        self.algorithm.reset()
        for _ in range(self.num_iter):
            observation, idx, weight = self.memory.sample_batch(self.batch_size)

            def closure(obs=observation, w=weight):
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses_ = self.algorithm(obs)
                loss = (losses_.loss * w.detach()).mean()
                loss.backward()
                #
                # torch.nn.utils.clip_grad_norm_(
                #     self.optimizer.param_groups[0]["params"], self.max_grad_norm
                # )
                return losses_

            if self.train_steps % self.policy_update_frequency == 0:
                cm = contextlib.nullcontext()
            else:
                cm = DisableGradient(self.policy)

            with cm:
                losses = self.optimizer.step(closure=closure)

            # Update memory
            self.memory.update(idx, losses.td_error.abs().detach())

            # Update logs
            self.logger.update(**average_named_tuple(losses)._asdict())
            self.logger.update(**self.algorithm.info())

            self.counters["train_steps"] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

            if self.early_stop(losses, **self.algorithm.info()):
                break

        self.algorithm.reset()
