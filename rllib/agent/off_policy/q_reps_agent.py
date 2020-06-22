"""Implementation of Q-REPS Agent."""
from rllib.algorithms.reps import QREPS
from rllib.policy.q_function_policy import SoftMax

from .reps_agent import REPSAgent


class QREPSAgent(REPSAgent):
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
        q_function,
        value_function,
        optimizer,
        memory,
        epsilon,
        regularization=True,
        batch_size=64,
        num_iter=1000,
        train_frequency=0,
        num_rollouts=1,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        if policy is None:
            policy = SoftMax(q_function, param=1.0 / epsilon)
        super().__init__(
            policy=policy,
            value_function=value_function,
            memory=memory,
            epsilon=epsilon,
            regularization=regularization,
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
        self.algorithm = QREPS(
            policy=policy,
            q_function=q_function,
            value_function=value_function,
            epsilon=epsilon,
            regularization=regularization,
            gamma=gamma,
        )
        # Over-write optimizer.
        self.optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **optimizer.defaults,
        )
        self.policy = self.algorithm.policy

    def _train(self):
        """See `AbstractAgent.train_agent'."""
        # old_policy = deep_copy_module(self.policy)
        self._optimizer_dual()
        # self.policy.prior = old_policy

        self.memory.reset()  # Erase memory.
        self.algorithm.update()  # Step the etas in REPS.
