"""Implementation of REPS Agent."""

from rllib.algorithms.reps import REPS
from rllib.util.neural_networks.utilities import deep_copy_module

from .off_policy_agent import OffPolicyAgent


class REPSAgent(OffPolicyAgent):
    """Implementation of the REPS algorithm.

    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self,
        policy,
        value_function,
        optimizer,
        memory,
        epsilon,
        batch_size,
        num_iter,
        regularization=False,
        train_frequency=0,
        num_rollouts=1,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        self.algorithm = REPS(
            policy=policy,
            value_function=value_function,
            epsilon=epsilon,
            regularization=regularization,
            gamma=gamma,
        )
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **optimizer.defaults,
        )

        super().__init__(
            memory=memory,
            batch_size=batch_size,
            optimizer=optimizer,
            num_iter=num_iter,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )

        self.policy = self.algorithm.policy

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
        self.algorithm.update()  # Step the etas in REPS.

    def _optimizer_dual(self):
        """Optimize the dual function."""
        self._optimize_loss(self.num_iter, loss_name="dual")

    def _fit_policy(self):
        """Fit the policy optimizing the weighted negative log-likelihood."""
        self._optimize_loss(self.num_iter, loss_name="policy_loss")

    def _optimize_loss(self, num_iter, loss_name="dual"):
        """Optimize the loss performing `num_iter' gradient steps."""
        for i in range(num_iter):
            obs, idx, weight = self.memory.sample_batch(self.batch_size)

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses = self.algorithm(
                    obs.state, obs.action, obs.reward, obs.next_state, obs.done
                )
                self.optimizer.zero_grad()
                loss_ = getattr(losses, loss_name)
                loss_.backward()
                return loss_

            loss = self.optimizer.step(closure=closure).item()
            self.logger.update(**{loss_name: loss})

            self.counters["train_steps"] += 1
