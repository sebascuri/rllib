"""Q Learning Algorithm."""

import torch

from rllib.policy import EpsGreedy

from .abstract_algorithm import AbstractAlgorithm


class QLearning(AbstractAlgorithm):
    r"""Implementation of Q-Learning algorithm.

    Q-Learning is an off-policy model-free control algorithm.

    The Q-Learning algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma \max_a Q(s', a)

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma \max_a Q(s', a)
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
    Watkins, C. J. C. H. (1989).
    Learning from delayed rewards.

    Watkins, C. J., & Dayan, P. (1992).
    Q-learning. Machine learning.

    Jaakkola, T., Jordan, M. I., & Singh, S. P. (1994).
    Convergence of stochastic iterative dynamic programming algorithms. NIPS.

    Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning.
    Machine learning.

    Mnih, V., et. al. (2013).
    Playing atari with deep reinforcement learning. NIPS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            policy=EpsGreedy(kwargs.get("critic"), param=0), *args, **kwargs
        )

    def get_value_target(self, observation):
        """Get q function target."""
        next_v = self.critic(observation.next_state).max(dim=-1)[0]
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v


class GradientQLearning(QLearning):
    r"""Implementation of Gradient Q Learning algorithm.

    The gradient q-learning algorithm propagates the gradient on both prediction and
    target estimates.

    .. math:: Q_{target} = (r(s, a) + \gamma \max_a Q(s', a))

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992).
    Q-learning. Machine learning.
    """

    def get_value_target(self, observation):
        """Get q function target."""
        with torch.enable_grad():  # Require gradient after it's been disabled.
            return super().get_value_target(observation)
