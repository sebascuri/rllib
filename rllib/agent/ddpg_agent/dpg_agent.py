"""Implementation of DPG Algorithm."""
from .abstract_dpg_agent import AbstractDPGAgent


class DPGAgent(AbstractDPGAgent):
    """Implementation of the DPG Algorithm.

    The DPG algorithm is a DPG algorithm with a full gradients.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * Q(x', \pi(x'))
        next_policy_action = self.policy(next_state).loc
        next_q = self.q_function(next_state, next_policy_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q
