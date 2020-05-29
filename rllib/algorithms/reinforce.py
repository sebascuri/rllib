"""REINFORCE Algorithm."""

import torch

from rllib.util.neural_networks import deep_copy_module, update_parameters
from rllib.util.utilities import tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm, PGLoss
from .gae import GAE


class REINFORCE(AbstractAlgorithm):
    r"""Implementation of REINFORCE algorithm.

    REINFORCE is an on-policy model-free control algorithm.
    REINFORCE computes the policy gradient using MC sample for the returns (sum of
    discounted rewards).


    The REINFORCE algorithm is a policy gradient algorithm that estimates the gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) \sum_{t' \geq t} r_{t'}

    Parameters
    ----------
    policy: AbstractPolicy
        Policy to optimize.
    baseline: AbstractValueFunction
        Baseline to reduce the variance of the gradient.
    criterion: _Loss
        Criterion to optimize the baseline.
    gamma: float
        Discount factor.

    References
    ----------
    Williams, Ronald J. (1992)
    Simple statistical gradient-following algorithms for connectionist reinforcement
    learning. Machine learning.
    """

    eps = 1e-12

    def __init__(self, policy, baseline, criterion, gamma):
        super().__init__()
        # Actor
        self.policy = policy
        self.baseline = baseline
        self.baseline_target = deep_copy_module(baseline)
        self.criterion = criterion
        self.gamma = gamma

        self.gae = GAE(1, self.gamma, self.baseline)

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        return self.gae(trajectory)  # GAE returns.

    def forward(self, trajectories):
        """Compute the losses."""
        actor_loss = torch.tensor(0.)
        baseline_loss = torch.tensor(0.)

        for trajectory in trajectories:
            state, action, reward, next_state, done, *r = trajectory

            # ACTOR LOSS
            pi = tensor_to_distribution(self.policy(state))
            if self.policy.discrete_action:
                action = action.long()
            with torch.no_grad():
                returns = self.returns(trajectory)
                (returns - returns.mean()) / (returns.std() + self.eps)

            actor_loss += (-pi.log_prob(action) * returns.detach()).sum()

            # BASELINE LOSS
            if self.baseline is not None:
                with torch.no_grad():
                    next_v = self.baseline_target(next_state)
                    target_v = reward + self.gamma * next_v * (1 - done)

                baseline_loss += self.criterion(self.baseline(state), target_v)

        return PGLoss(actor_loss + baseline_loss, actor_loss, baseline_loss)

    def update(self):
        """Update the baseline network."""
        if self.baseline is not None:
            update_parameters(self.baseline_target, self.baseline, self.baseline.tau)
