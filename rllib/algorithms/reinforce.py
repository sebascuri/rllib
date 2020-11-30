"""REINFORCE Algorithm."""
from .gaac import GAAC


class REINFORCE(GAAC):
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
    critic: AbstractValueFunction
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

    def __init__(self, *args, **kwargs):
        super().__init__(lambda_=1, *args, **kwargs)
