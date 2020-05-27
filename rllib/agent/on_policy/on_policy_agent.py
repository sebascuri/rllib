"""On Policy Agent."""

from rllib.agent.abstract_agent import AbstractAgent

from rllib.dataset.utilities import stack_list_of_tuples, average_named_tuple


class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, env_name, optimizer,
                 batch_size=1, target_update_frequency=1, num_iter=1,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name, train_frequency=train_frequency,
                         num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.trajectories = []
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self.num_iter = num_iter

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)
        if (self._training and  # training mode.
                len(self.trajectories[-1]) >= self.batch_size and  # enough data.
                self.train_frequency > 0 and  # train after train_frequency transitions.
                self.total_steps % self.train_frequency == 0):  # correct steps.
            self._train()
            self.trajectories[-1] = []

    def start_episode(self, **kwargs):
        """See `AbstractAgent.start_episode'."""
        super().start_episode(**kwargs)
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (self._training and  # training mode.
                self.num_rollouts > 0 and  # train after num_rollouts transitions.
                self.total_episodes % self.num_rollouts == 0):  # correct steps.
            self._train()
            self.trajectories = list()
        if self.num_rollouts == 0:
            self.trajectories = list()

        super().end_episode()

    def _train(self):
        trajectories = [stack_list_of_tuples(t) for t in self.trajectories]
        """Train Policy Gradient Agent."""
        for _ in range(self.num_iter):
            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses = self.algorithm(trajectories)
                losses.loss.backward()
                return losses

            losses = self.optimizer.step(closure=closure)
            # Update logs
            self.logger.update(**average_named_tuple(losses)._asdict())
            self.logger.update(**self.algorithm.info())

            self.counters['train_steps'] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()
