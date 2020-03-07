"""REINFORCE Algorithm."""
import torch
import torch.nn as nn

from rllib.util import discount_cumsum
from collections import namedtuple

REINFORCELoss = namedtuple('REINFORCELoss',
                           ['actor_loss', 'baseline_loss'])


class REINFORCE(nn.Module):
    r"""Implementation of REINFORCE algorithm.

    REINFORCE is an on-policy model-free control algorithm.
    REINFORCE computes the policy gradient using MC sample for the returns (sum of
    discounted rewards).


    The REINFORCE algorithm is a policy gradient algorithm that estimates the gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) \sum_{t' \geq t} r_{t'}

    Parameters
    ----------
    policy: AbstractPolicy
        policy to optimize.
    baseline: AbstractValueFunction
        baseline to reduce the variance of the gradient.
    criterion: _Loss
        Criterion to optimize the baseline.
    gamma: float
        discount factor.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

    """

    eps = 1e-12

    def __init__(self, policy, baseline, criterion, gamma):
        super().__init__()
        # Actor
        self.policy = policy
        self.baseline = baseline
        self.criterion = criterion
        self.gamma = gamma

    def _value_estimate(self, trajectory):
        val = discount_cumsum(trajectory.reward, self.gamma)
        return (val - val.mean()) / (val.std() + self.eps)

    def forward(self, trajectories):
        """Compute the losses."""
        actor_loss = torch.tensor(0.)
        baseline_loss = torch.tensor(0.)

        for trajectory in trajectories:
            value_estimate = self._value_estimate(trajectory)

            if self.baseline is not None:
                baseline = self.baseline(trajectory.state)
            else:
                baseline = torch.zeros_like(trajectory.reward)

            pi = self.policy(trajectory.state)
            action = trajectory.action
            if self.policy.discrete_action:
                action = trajectory.action.long()
            actor_loss += (
                    - pi.log_prob(action) * (value_estimate - baseline.detach())).sum()

            if self.baseline is not None:
                baseline_loss += self.criterion(baseline, value_estimate).mean()

        return REINFORCELoss(actor_loss, baseline_loss)
