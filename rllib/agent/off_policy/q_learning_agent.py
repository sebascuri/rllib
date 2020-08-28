"""Implementation of QLearning Algorithms."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.q_learning import QLearning
from rllib.dataset.experience_replay import ExperienceReplay
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
    q_function: AbstractQFunction
        q_function that is learned.
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

    def __init__(self, q_function, policy, criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.algorithm = QLearning(
            critic=q_function, criterion=criterion(reduction="none"), gamma=self.gamma
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction.default(environment)
        policy = EpsGreedy(q_function, ExponentialDecay(1.0, 0.01, 500))
        optimizer = Adam(q_function.parameters(), lr=3e-4)
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        return cls(
            q_function=q_function,
            policy=policy,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            num_iter=1,
            batch_size=100,
            target_update_frequency=1,
            train_frequency=1,
            num_rollouts=0,
            comment=environment.name,
            *args,
            **kwargs,
        )
