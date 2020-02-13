"""Implementation of DDPG Algorithm."""
from .abstract_dpg_agent import AbstractDPGAgent


class DDPGAgent(AbstractDPGAgent):
    """Implementation of the DDPG Algorithm.

    The DDPG algorithm is a DPG algorithm with a delayed target q function.

    References
    ----------
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        # target = r + gamma * Q_target(x', \pi_target(x'))
        next_policy_action = self.policy_target(next_state).loc
        next_q = self.q_target(next_state, next_policy_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()
