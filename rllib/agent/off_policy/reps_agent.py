"""Implementation of REPS Agent."""

from torch.optim import Adam

from rllib.algorithms.reps import REPS
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.util.neural_networks.utilities import deep_copy_module
from rllib.value_function import NNValueFunction

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
            observation, idx, weight = self.memory.sample_batch(self.batch_size)

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses_ = self.algorithm(observation)
                self.optimizer.zero_grad()
                loss_ = getattr(losses_, loss_name)
                loss_.backward()
                return loss_

            losses = self.optimizer.step(closure=closure).item()
            self.logger.update(**{loss_name: losses})

            self.counters["train_steps"] += 1
            if self._early_stop_training(losses, **self.algorithm.info()):
                break

    @classmethod
    def default(
        cls,
        environment,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
        """See `AbstractAgent.default'."""
        value_function = NNValueFunction(
            dim_state=environment.dim_state,
            num_states=environment.num_states,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            tau=5e-3,
            input_transform=None,
        )
        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            tau=5e-3,
            input_transform=None,
            deterministic=False,
        )

        optimizer = Adam(value_function.parameters(), lr=3e-4)
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        return cls(
            policy=policy,
            value_function=value_function,
            optimizer=optimizer,
            memory=memory,
            epsilon=1.0,
            regularization=False,
            num_iter=5 if test else 200,
            batch_size=100,
            train_frequency=0,
            num_rollouts=15,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
