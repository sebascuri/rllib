"""Implementation of delayed SARSA Algorithm."""

from .abstract_sarsa_agent import AbstractSARSAAgent


class DSARSAAgent(AbstractSARSAAgent):
    """Implementation of Delayed-SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action):
        pred_q = self.q_function(state, action)
        next_v = self.q_target(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)

        return pred_q, target_q.detach()
