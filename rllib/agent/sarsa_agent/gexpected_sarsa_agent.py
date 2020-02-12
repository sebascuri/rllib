"""Implementation of semi-gradient expected SARSA Algorithm."""

from .abstract_sarsa_agent import AbstractSARSAAgent
import torch


class GExpectedSARSAAgent(AbstractSARSAAgent):
    """Implementation of Delayed-SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action):
        pred_q = self.q_function(state, action)
        probs = self.policy(state).probs
        target_q = torch.tensor(0.)
        for na, p in enumerate(probs):
            target_q += p * self.q_function(next_state, torch.tensor(na))

        target_q = reward + self.gamma * target_q * (1 - done)

        return pred_q, target_q.detach()
