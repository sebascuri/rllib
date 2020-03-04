"""Implementation of expected SARSA Algorithm."""

from .abstract_sarsa_agent import AbstractSARSAAgent


class ExpectedSARSAAgent(AbstractSARSAAgent):
    """Implementation of Delayed-SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action, *args, **kwargs
            ):
        pred_q = self.q_function(state, action)
        next_v = self.q_function.value(next_state, self.policy)
        target_q = reward + self.gamma * next_v * (1 - done)

        return pred_q, target_q
