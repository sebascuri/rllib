"""Generalized Advantage Estimation Algorithm."""

import torch.nn as nn

from rllib.util.neural_networks.utilities import broadcast_to_tensor
from rllib.util.utilities import RewardTransformer
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
    td_lambda: float
        Eligibility trace parameter.
    gamma: float
        Discount factor.

    References
    ----------
    Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
    High-dimensional continuous control using generalized advantage estimation. ICLR.
    """

    def __init__(
        self,
        td_lambda,
        gamma,
        reward_transformer=RewardTransformer(),
        value_function=None,
    ):
        super().__init__()
        if value_function is None:
            assert (
                td_lambda == 1
            ), "If no value function is given, then lambda must be 1."
        self.value_function = value_function
        self.lambda_gamma = td_lambda * gamma
        self.reward_transformer = reward_transformer

    def forward(self, observation):
        """Compute the GAE estimation."""
        state, action, reward, next_state, done, *r = observation
        reward = self.reward_transformer(reward)
        if self.value_function is None:
            td_error = reward
        else:
            next_v = self.value_function(next_state)
            not_done = broadcast_to_tensor(1.0 - done, target_tensor=next_v)
            next_v = next_v * not_done
            td_error = reward + next_v - self.value_function(state)

        return discount_cumsum(td_error, self.lambda_gamma)
