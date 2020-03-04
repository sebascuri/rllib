"""Implementation of DDPG Algorithm."""
from .abstract_dpg_agent import AbstractDPGAgent
import torch


class DDPGAgent(AbstractDPGAgent):
    """Implementation of the DDPG Algorithm.

    The DDPG algorithm is a DPG algorithm with a delayed target q function.

    References
    ----------
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def _td(self, state, action, reward, next_state, done, *args, **kwargs):
        pred_q = self.q_function(state, action)

        # target = r + gamma * Q_target(x', \pi_target(x') + noise)
        next_policy_action = self.policy_target(next_state).mean
        next_action_noise = (torch.randn_like(next_policy_action) * self.policy_noise
                             ).clamp(-self.noise_clip, self.noise_clip)
        next_policy_action = (next_policy_action + next_action_noise).clamp(-1, 1)

        next_v = self.q_target(next_state, next_policy_action)
        if type(next_v) is list:
            next_v = torch.min(*next_v)
        target_q = reward + self.gamma * next_v * (1 - done)

        return pred_q, target_q.detach()
