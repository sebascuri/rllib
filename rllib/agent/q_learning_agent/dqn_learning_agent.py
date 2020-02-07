"""Implementation of DQN Algorithm."""
from .abstract_q_learning_agent import AbstractQLearningAgent


class DQNAgent(AbstractQLearningAgent):
    """Implementation of DQN algorithm.

    loss = l[Q(x, a), r + max_a Q'(x', a)]

    References
    ----------
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529.
    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q_target(x', a)
        next_q = self.q_target.max(next_state)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
