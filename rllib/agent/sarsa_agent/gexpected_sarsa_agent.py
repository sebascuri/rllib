"""Implementation of semi-gradient expected SARSA Algorithm."""

from .abstract_sarsa_agent import AbstractSARSAAgent
import torch


class GExpectedSARSAAgent(AbstractSARSAAgent):
    """Implementation of Delayed-SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action):
        pred_q = self.q_function(state, action)
        probs = self.policy(state).probs

        batch_size = state.shape[0]
        target_q = torch.zeros(batch_size)
        for na in range(self.policy.num_actions):
            na = torch.tensor([na]).repeat(batch_size, 1)
            target_q += probs.gather(-1, na).squeeze() * self.q_function(
                next_state, na.squeeze(-1))

        target_q = reward + self.gamma * target_q * (1 - done)

        return pred_q, target_q.detach()
