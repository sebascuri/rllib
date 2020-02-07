"""Implementation of Q-Learning Algorithm."""
from .abstract_q_learning_agent import AbstractQLearningAgent


class QLearningAgent(AbstractQLearningAgent):
    """Implementation of Q-Learning algorithm.

    loss = l[Q(x, a), r + Q(x', arg max Q(x', a))]

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and don't stop gradient.
        target_q = self.q_function.max(next_state)
        target_q = reward + self.gamma * target_q * (1 - done)

        return pred_q, target_q
