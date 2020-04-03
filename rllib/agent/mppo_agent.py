"""MPPO Agent Implementation."""

from rllib.agent.abstract_agent import AbstractAgent
from torch.utils.data import DataLoader


class MPPOAgent(AbstractAgent):
    """Implementation of an agent that runs MPPO."""

    def __init__(self, environment, mppo, optimizer, memory, num_rollouts=1,
                 num_iter=100, batch_size=64, target_update_frequency=4,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(environment, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)

        self.mppo = mppo
        self.policy = mppo.policy
        self.optimizer = optimizer
        self.memory = memory
        self.target_update_frequency = target_update_frequency

        self.num_rollouts = num_rollouts
        self.num_iter = num_iter
        self.batch_size = batch_size

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == self.num_rollouts - 1:
            if self._training:
                self._train()
                self.memory.reset()
        super().end_episode()

    def _train(self) -> None:
        self.mppo.reset()
        loader = DataLoader(self.memory, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.num_iter):
            for obs, idx, weights in loader:
                self.optimizer.zero_grad()
                losses = self.mppo(obs.state, obs.action, obs.reward, obs.next_state,
                                   obs.done)
                losses.loss.backward()
                self.optimizer.step()

                self.logger.update(**losses._asdict())
