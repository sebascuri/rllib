"""Implementation of QLearning Algorithms."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.q_learning import QLearning
from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class QLearningAgent(OffPolicyAgent):
    """Implementation of a Q-Learning agent.

    The Q-Learning algorithm implements the Q-Learning algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.

    """

    def __init__(self, critic, policy, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = QLearning(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(
        cls,
        environment,
        critic=None,
        policy=None,
        epsilon=None,
        lr=3e-4,
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNQFunction.default(environment)

        if policy is None:
            if epsilon is None:
                epsilon = ExponentialDecay(1.0, 0.01, 500)
            policy = EpsGreedy(critic, epsilon)
        optimizer = Adam(critic.parameters(), lr=lr)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
