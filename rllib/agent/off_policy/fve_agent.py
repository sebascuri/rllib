"""Fitted value evaluation agent."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.fitted_value_evaluation import FittedValueEvaluationAlgorithm
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class FittedValueEvaluationAgent(OffPolicyAgent):
    """Fitted value evaluation agent.

    It just evaluates the critic collecting data using fitted td-learning.

    References
    ----------
    Munos, R., & Szepesvári, C. (2008).
    Finite-Time Bounds for Fitted Value Iteration. JMLR.
    """

    def __init__(self, critic, policy, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = FittedValueEvaluationAlgorithm(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, policy, critic=None, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNQFunction.default(environment)
        optimizer = Adam(critic.parameters(), lr=lr)
        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
