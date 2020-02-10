"""Implementation of GQ-Learning Algorithms."""
from .q_learning_agent import QLearningAgent


class GQLearningAgent(QLearningAgent):
    """Implementation of Gradient Q-Learning algorithm.

    loss = l[Q(x, a), r + Q(x', arg max Q(x', a)).stop_gradient]

    References
    ----------
    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q, target_q = super()._td(state, action, reward, next_state, done)
        return pred_q, target_q.detach()
