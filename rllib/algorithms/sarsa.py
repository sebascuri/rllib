"""SARSA Algorithm."""

import torch

from rllib.policy import EpsGreedy

from .abstract_algorithm import AbstractAlgorithm


class SARSA(AbstractAlgorithm):
    r"""Implementation of SARSA algorithm.

    SARSA is an on-policy model-free control algorithm.

    The SARSA algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma Q(s', a')
    where a' is sampled from a greedy policy w.r.t the current Q-Value estimate.

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma Q(s', a')
    .. math:: \mathcal{L}(Q(s, a), Q_{target})

    Parameters
    ----------
    critic: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Rummery, G. A., & Niranjan, M. (1994).
    On-line Q-learning using connectionist systems. Cambridge, UK.

    Sutton, R. S. (1996).
    Generalization in reinforcement learning: Successful examples using sparse coarse
    coding. NIPS.

    Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000).
    Convergence results for single-step on-policy reinforcement-learning algorithms.
    Machine learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            policy=kwargs.pop("policy", EpsGreedy(kwargs.get("critic"), 0)),
            *args,
            **kwargs,
        )

    def get_value_target(self, observation):
        """Get q function target."""
        next_v = self.critic_target(observation.next_state, observation.next_action)
        next_v = next_v * (1 - observation.done)
        return self.get_reward(observation) + self.gamma * next_v


class GradientSARSA(SARSA):
    r"""Implementation of Gradient SARSA.

    The gradient SARSA algorithm takes the gradient of the target value too.

    .. math:: Q_{target} = (r(s, a) + \gamma Q(s', a')).detach()

    References
    ----------
    TODO: find
    """

    def get_value_target(self, observation):
        """Get q function target."""
        with torch.enable_grad():  # Require gradient after it's been disabled.
            return super().get_value_target(observation)
