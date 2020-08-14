"""Q Learning Algorithm."""

import torch

from rllib.policy import EpsGreedy
from rllib.util.neural_networks import deep_copy_module, update_parameters

from .abstract_algorithm import AbstractAlgorithm, TDLoss


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
    q_function: AbstractQFunction
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

    def __init__(self, q_function, criterion, *args, **kwargs):
        super().__init__(policy=EpsGreedy(q_function, param=0), *args, **kwargs)
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.criterion = criterion

    def get_q_target(self, observation):
        """Get q function target."""
        next_v = self.q_function(observation.next_state).max(dim=-1)[0]
        next_v = next_v * (1 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        pred_q = self.q_function(state, action)
        with torch.no_grad():
            target_q = self.get_q_target(observation)

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return TDLoss(
            loss=self.criterion(pred_q, target_q).squeeze(-1),
            td_error=(pred_q - target_q).detach().squeeze(-1),
        )

    def update(self):
        """Update the target network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


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

    def get_q_target(self, observation):
        """Get q function target."""
        with torch.enable_grad():  # Require gradient after it's been disabled.
            return super().get_q_target(observation)
