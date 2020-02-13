"""Implementation of GDPG Algorithm."""
from .dpg_agent import DPGAgent


class GDPGAgent(DPGAgent):
    """Implementation of the GDPG Algorithm.

    The GDPG algorithm is a DPG algorithm with a semi-gradients.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q, target_q = super()._td(state, action, reward, next_state, done)
        return pred_q, target_q.detach()
