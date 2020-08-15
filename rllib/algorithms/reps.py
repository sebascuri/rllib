"""Implementation of REPS Algorithm."""

import torch
import torch.distributions

from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm, Loss


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

    def __init__(self, epsilon, regularization=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if regularization:
            eta = epsilon
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self.eta = eta
            self.epsilon = torch.tensor(0.0)
        else:
            self.eta = Learnable(1.0)
            self.epsilon = torch.tensor(epsilon)

    def _project_eta(self):
        """Project the etas to be positive inplace."""
        # Since we divide by eta, make sure it doesn't go to zero.
        self.eta.start.data.clamp_(min=1e-5)

    def _policy_weighted_nll(self, state, action, weights):
        """Return weighted policy negative log-likelihood."""
        action_log_probs = tensor_to_distribution(self.policy(state)).log_prob(action)
        weighted_log_prob = weights.detach() * action_log_probs

        # Clamping is crucial for stability so that it does not converge to a delta.
        log_likelihood = torch.mean(weighted_log_prob.clamp_max(1e-3))
        return -log_likelihood

    def get_value_target(self, observation):
        """Get value-function target."""
        next_v = self.critic(observation.next_state) * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def forward_slow(self, observation):
        """Return primal and dual loss terms from REPS."""
        state, action, reward, next_state, done, *r = observation
        # Make sure the lagrange multipliers stay positive.
        self._project_eta()

        # Compute Scaled TD-Errors
        value = self.critic(state)

        # For dual function we need the full gradient, not the semi gradient!
        target = self.get_value_target(observation)
        td = target - value

        weights = td / self.eta()
        normalizer = torch.logsumexp(weights, dim=0)
        dual = self.eta() * (self.epsilon + normalizer) + (1.0 - self.gamma) * value

        nll = self._policy_weighted_nll(state, action, weights)

        return Loss(
            loss=dual.mean() + nll, dual_loss=dual.mean(), policy_loss=nll, td_error=td
        )

    def update(self):
        """Update regularization parameter."""
        super().update()
        self.eta.update()
