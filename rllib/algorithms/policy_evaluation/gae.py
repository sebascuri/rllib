"""Generalized Advantage Estimation Algorithm."""

import torch.nn as nn

from rllib.util.value_estimation import discount_cumsum


class GAE(nn.Module):
    r"""Implementation of Generalized Advantage Estimation Algorithm.

    Generalized Advantage Estimation is an on-policy policy evaluation algorithm.

    Generalized Advantage Estimation estimates the advantage function as:
    .. math:: A(s, a) = \sum_t (\gamma \lambda)^t \delta_t,
    where the td error is:
    .. math:: \delta_t = r + \gamma V_t(s_{t+1}) - V(s_t)

    It has a parameter, lambda, that interpolates between the REINFORCE estimate,
    when lambda = 1, and the TD-Residual estimate, when lambda = 0.


    Parameters
    ----------
    value_function: AbstractValueFunction
        value function to reduce evaluate_agent variance the gradient.
    lambda_: float
        Eligibility trace parameter.
    gamma: float
        Discount factor.

    References
    ----------
    Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
    High-dimensional continuous control using generalized advantage estimation. ICLR.
    """

    def __init__(self, lambda_, gamma, value_function=None):
        super().__init__()
        if value_function is None:
            assert lambda_ == 1, "If no value function is given, then lambda must be 1."
        self.value_function = value_function
        self.lambda_gamma = lambda_ * gamma

    def forward(self, observation):
        """Compute the GAE estimation."""
        state, action, reward, next_state, done, *r = observation
        if self.value_function is None:
            td_error = observation.reward
        else:
            next_v = self.value_function(next_state) * (1 - done)
            td_error = reward + next_v - self.value_function(state)

        return discount_cumsum(td_error, self.lambda_gamma)
