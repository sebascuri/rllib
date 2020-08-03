"""DDQN Algorithm."""

from .q_learning import QLearning


class DDQN(QLearning):
    r"""Implementation of Double Delayed Q Learning algorithm.

    The double q-learning algorithm calculates the target value with the action that
    maximizes the primal function to mitigate over-estimation bias.

    a_{target} = \arg max_a Q(s', a)
    Q_{target} = (r(s, a) + \gamma \max_a Q_{target}(s', a_{target})).detach()

    References
    ----------
    Hasselt, H. V. (2010).
    Double Q-learning. NIPS.

    Van Hasselt, Hado, Arthur Guez, and David Silver. (2016)
    Deep reinforcement learning with double q-learning. AAAI.
    """

    def get_target(self, reward, next_state, done):
        """Get q function target."""
        next_action = self.q_function(next_state).argmax(dim=-1)
        next_v = self.q_target(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)
        return target_q
