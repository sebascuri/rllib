"""Implementation of REPS Agent."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.util.neural_networks.utilities import deep_copy_module


class REPSAgent(AbstractAgent):
    """Implementation of the REPS algorithm.

    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(self, env_name, reps_loss, optimizer, memory,
                 num_rollouts, batch_size, num_dual_iter, num_policy_iter=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.policy = reps_loss.policy
        self.reps = reps_loss

        self.optimizer = optimizer
        self.memory = memory
        self.num_dual_iter = num_dual_iter
        self.num_policy_iter = num_policy_iter
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
        old_policy = deep_copy_module(self.policy)
        self._optimizer_dual()

        self.policy.prior = old_policy
        self._fit_policy()

        self.memory.reset()  # Erase memory.
        self.reps.update_eta()  # Step the etas in REPS.

    def _optimizer_dual(self):
        """Optimize the dual function."""
        self._optimize_loss(self.num_dual_iter, loss_name='dual')

    def _fit_policy(self):
        """Fit the policy optimizing the weighted negative log-likelihood."""
        self._optimize_loss(self.num_policy_iter, loss_name='policy_nll')

    def _optimize_loss(self, num_iter, loss_name='dual'):
        """Optimize the loss performing `num_iter' gradient steps."""
        for i in range(num_iter):
            obs, idx, weight = self.memory.get_batch(self.batch_size)

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses = self.reps(obs.state, obs.action, obs.reward, obs.next_state,
                                   obs.done)
                self.optimizer.zero_grad()
                loss_ = getattr(losses, loss_name)
                loss_.backward()
                return loss_

            loss = self.optimizer.step(closure=closure).item()
            self.logger.update(**{loss_name: loss})
