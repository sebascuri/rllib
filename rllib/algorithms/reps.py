"""Implementation of REPS Algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import get_entropy_and_log_p, tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm


class REPSLoss(nn.Module):
    """REPS Loss."""

    def __init__(self, epsilon=0.1, relent_regularization=False):
        super().__init__()

        if relent_regularization:
            eta = epsilon
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)

            self._eta = eta

            self.epsilon = torch.tensor(0.0)

        else:  # Trust-Region: || KL(p || q) || < \epsilon
            self._eta = Learnable(1.0, positive=True)
            self.epsilon = torch.tensor(epsilon)

    @property
    def eta(self):
        """Get REPS regularization parameter."""
        return self._eta().detach()

    def forward(self, action_log_p, value, target):
        """Return primal and dual loss terms from REPS.

        Parameters
        ----------
        action_log_p : torch.Tensor
            A [state_batch, 1] tensor of log probabilities of the corresponding actions
            under the policy.
        value: torch.Tensor
            The value function (with gradients) evaluated at V(s)
        target: torch.Tensor
            The value target (with gradients) evaluated at r + gamma V(s')
        """
        td = target - value
        weights = td / self._eta()
        normalizer = torch.logsumexp(weights, dim=0)
        dual_loss = self._eta() * (self.epsilon + normalizer)

        # Clamping is crucial for stability so that it does not converge to a delta.
        weighted_log_p = torch.exp(weights).clamp_max(1e2).detach() * action_log_p
        log_likelihood = weighted_log_p.mean()

        return Loss(
            policy_loss=-log_likelihood, dual_loss=dual_loss, td_error=td.mean()
        )


class REPS(AbstractAlgorithm):
    r"""Relative Entropy Policy Search Algorithm.

    REPS optimizes the following regularized LP over the set of distributions \mu(X, A).

    ..math::  \max \mu r - eta R(\mu, d_0)
    ..math::  s.t. \sum_a \mu(x, a) = \sum_{x', a'} = \mu(x', a') P(x|x', a'),

    where R is the relative entropy between \mu and any distribution d.
    This differs from the original formulation in which R(\mu, d) is used to express a
    trust region.

    The dual of the LP is:
    ..math::  G(V) = \eta \log \sum_{x, a} d_0(x, a) \exp^{\delta(x, a) / \eta}
    where \delta(x,a) = r + \sum_{x'} P(x'|x, a) V(x') - V(x) is the TD-error and V(x)
    are the dual variables associated with the stationary constraints in the primal.
    V(x) is usually referred to as the value function.

    Using d(x,a) as the empirical distribution, G(V) can be approximated by samples.

    The optimal policy is given by:
    ..math::  \pi(a|x) \propto d_0(x, a) \exp^{\delta(x, a) / \eta}.

    Instead of setting the policy to \pi(a|x) at sampled (x, a), we can fit the policy
    by minimizing the negative log-likelihood at the sampled elements.


    Calling REPS() returns a sampled based estimate of G(V) and the NLL of the policy.
    Both G(V) and NLL lend are differentiable and lend themselves to gradient based
    optimization.


    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self, reps_eta, relent_regularization=False, learn_policy=True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learn_policy = learn_policy
        self.reps_loss = REPSLoss(
            epsilon=reps_eta, relent_regularization=relent_regularization
        )

    def critic_loss(self, observation) -> Loss:
        """Get the critic loss."""
        return Loss()

    def get_value_target(self, observation):
        """Get value-function target."""
        next_v = self.critic(observation.next_state) * (1 - observation.done)
        return self.get_reward(observation) + self.gamma * next_v

    def actor_loss(self, observation):
        """Return primal and dual loss terms from REPS."""
        state, action, reward, next_state, done, *r = observation

        # Compute Scaled TD-Errors
        value = self.critic(state)

        # For dual function we need the full gradient, not the semi gradient!
        target = self.get_value_target(observation)

        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        _, action_log_p = get_entropy_and_log_p(pi, action, self.policy.action_scale)

        reps_loss = self.reps_loss(action_log_p, value, target)
        self._info.update(reps_eta=self.reps_loss.eta)
        return reps_loss + Loss(dual_loss=(1.0 - self.gamma) * value.mean())
