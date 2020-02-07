"""Implementation of DDQN Algorithm."""
from .abstract_q_learning_agent import AbstractQLearningAgent


class DDQNAgent(AbstractQLearningAgent):
    """Implementation of Double DQN algorithm.

    loss = l[Q(x, a), r + Q'(x', argmax Q(x,a))]

    References
    ----------
    Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning
    with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * Q_target(x', argmax Q(x', a))

        next_action = self.q_function.argmax(next_state)
        next_q = self.q_target(next_state, next_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
