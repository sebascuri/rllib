"""Off Policy Agent."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import average_named_tuple


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(
        self,
        memory,
        optimizer,
        target_update_frequency=1,
        num_iter=1,
        batch_size=64,
        train_frequency=0,
        num_rollouts=1,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
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
            and len(self.memory) >= self.batch_size  # enough data.
            and self.train_frequency > 0  # train after a transition.
            and self.total_steps % self.train_frequency == 0  # correct steps.
        ):  # correct steps.
            self._train()

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (
            self._training  # training mode.
            and len(self.memory) > self.batch_size  # enough data.
            and self.num_rollouts > 0  # train once the episode ends.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        ):  # use total_episodes + 1 because the super() is called after training.
            self._train()

        if len(self.memory) > 0:  # Maybe _train() resets the memory.
            self.memory.end_episode()

        super().end_episode()  # this update total episodes.

    def _train(self):
        """Train the off-policy agent."""
        self.algorithm.reset()
        for _ in range(self.num_iter):
            obs, idx, weight = self.memory.sample_batch(self.batch_size)

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses_ = self.algorithm(
                    obs.state, obs.action, obs.reward, obs.next_state, obs.done
                )
                loss = (losses_.loss * weight.detach()).mean()
                loss.backward()
                return losses_

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

        self.algorithm.reset()
