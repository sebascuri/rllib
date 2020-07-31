"""Implementation of Q-REPS Agent."""
from torch.optim import Adam

from rllib.algorithms.reps import QREPS
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.policy.q_function_policy import SoftMax
from rllib.value_function import NNQFunction, NNValueFunction

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
        gamma=0.99,
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

    def learn(self):
        """See `AbstractAgent.train_agent'."""
        # old_policy = deep_copy_module(self.policy)
        self._optimizer_dual()
        # self.policy.prior = old_policy

        self.memory.reset()  # Erase memory.
        self.algorithm.update()  # Step the etas in REPS.

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
        q_function = NNQFunction(
            environment.dim_state,
            environment.dim_action,
            environment.num_states,
            environment.num_actions,
            layers=[64, 64],
        )

        if environment.num_actions > 0:
            policy = None
        else:
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
            q_function=q_function,
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
