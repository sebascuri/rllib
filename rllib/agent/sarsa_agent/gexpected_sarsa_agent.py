"""Implementation of semi-gradient expected SARSA Algorithm."""

from .expected_sarsa_agent import ExpectedSARSAAgent


class GExpectedSARSAAgent(ExpectedSARSAAgent):
    """Implementation of Delayed-SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action):
        pred_q, target_q = super()._td(state, action, reward, next_state, done,
                                       next_action)
        return pred_q, target_q.detach()
