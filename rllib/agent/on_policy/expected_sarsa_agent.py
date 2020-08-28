"""Implementation of Expected SARSA Agent."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.esarsa import ESARSA
from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

from .on_policy_agent import OnPolicyAgent


class ExpectedSARSAAgent(OnPolicyAgent):
    """Implementation of an Expected SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
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
    """

    def __init__(self, q_function, policy, criterion, *args, **kwargs):
        super().__init__(
            num_rollouts=kwargs.pop("num_rollouts", 0),
            train_frequency=kwargs.pop("train_frequency", 1),
            *args,
            **kwargs,
        )
        self.algorithm = ESARSA(
            critic=q_function,
            criterion=criterion(reduction="mean"),
            policy=policy,
            gamma=self.gamma,
        )
        self.policy = policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction.default(environment, tau=0)
        policy = EpsGreedy(q_function, ExponentialDecay(start=1.0, end=0.01, decay=500))
        optimizer = Adam(q_function.parameters(), lr=3e-4)
        criterion = loss.MSELoss

        return cls(
            q_function=q_function,
            policy=policy,
            optimizer=optimizer,
            criterion=criterion,
            num_iter=1,
            target_update_frequency=4,
            train_frequency=1,
            num_rollouts=0,
            comment=environment.name,
            *args,
            **kwargs,
        )
