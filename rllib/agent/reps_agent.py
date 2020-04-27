"""Implementation of REPS Agent."""
from torch.utils.data import DataLoader

from rllib.agent.abstract_agent import AbstractAgent


class REPSAgent(AbstractAgent):
    """Implementation of the REPS algorithm.

    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(self, environment, reps_loss, optimizer, memory,
                 num_iter, num_rollouts, batch_size,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(environment, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.policy = reps_loss.policy
        self.reps = reps_loss
        self.optimizer = optimizer
        self.memory = memory
        self.num_iter = num_iter
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (self.total_episodes + 1) % self.num_rollouts == 0:
            if self._training:
                self._train()

        super().end_episode()

    def _train(self):
        """See `AbstractAgent.train_agent'."""
        data_loader = DataLoader(self.memory, batch_size=self.batch_size, shuffle=True)
        self._optimizer_dual(data_loader)
        self._fit_policy(data_loader)
        self.memory.reset()  # Empty memory.
        self.reps.update_eta()  # Step the etas in REPS.

    def _optimizer_dual(self, data_loader):
        self._optimize_loss(data_loader, loss='dual')

    def _fit_policy(self, data_loader):
        self._optimize_loss(data_loader, loss='policy')

    def _optimize_loss(self, data_loader, loss='dual'):
        for i in range(self.num_iter):
            for obs, idx, weight in data_loader:
                losses = self.reps(obs.state, obs.action, obs.reward, obs.next_state,
                                   obs.done)
                self.optimizer.zero_grad()
                if loss == 'dual':
                    losses.dual.backward()
                elif loss == 'policy':
                    losses.policy_nll.backward()
                elif loss == 'combined':
                    (losses.dual + losses.policy_nll).backward()
                else:
                    raise NotImplementedError(f"{loss} not implemented.")
                self.optimizer.step()

                self.logger.update(**losses._asdict())
