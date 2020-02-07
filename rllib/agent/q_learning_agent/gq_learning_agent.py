"""Implementation of GQ-Learning Algorithms."""
from .abstract_q_learning_agent import AbstractQLearningAgent


class GQLearningAgent(AbstractQLearningAgent):
    """Implementation of Gradient Q-Learning algorithm.

    loss = l[Q(x, a), r + Q(x', arg max Q(x', a)).stop_gradient]

    References
    ----------
    Sutton, Richard S., et al. "Fast gradient-descent methods for temporal-difference
    learning with linear function approximation." Proceedings of the 26th Annual
    International Conference on Machine Learning. ACM, 2009.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and stop gradient.
        next_q = self.q_function.max(next_state)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
