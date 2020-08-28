"""Implementation of REINFORCE Algorithms."""

import torch.nn as nn
from torch.optim import Adam

from rllib.algorithms.reinforce import REINFORCE
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .on_policy_agent import OnPolicyAgent


class REINFORCEAgent(OnPolicyAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, policy, baseline=None, criterion=nn.MSELoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = REINFORCE(
            policy=policy,
            baseline=baseline,
            criterion=criterion(reduction="mean"),
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        policy = NNPolicy.default(environment)
        baseline = NNValueFunction.default(environment)

        optimizer = Adam(
            [
                {"params": policy.parameters(), "lr": 1e-4},
                {"params": baseline.parameters(), "lr": 1e-3},
            ]
        )
        criterion = nn.MSELoss

        return cls(
            policy=policy,
            baseline=baseline,
            optimizer=optimizer,
            criterion=criterion,
            num_iter=1,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=1,
            comment=environment.name,
            *args,
            **kwargs,
        )
