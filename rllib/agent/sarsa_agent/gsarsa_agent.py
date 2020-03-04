"""Implementation of semi-gradient SARSA Algorithm."""

from .sarsa_agent import SARSAAgent


class GSARSAAgent(SARSAAgent):
    """Implementation of semi-gradient SARSA (On-Line)-Control."""

    def _td(self, state, action, reward, next_state, done, next_action, *args, **kwargs
            ):
        pred_q, target_q = super()._td(state, action, reward, next_state, done,
                                       next_action)
        return pred_q, target_q.detach()
