"""Implementation of REINFORCE Algorithm."""

from .abstract_policy_gradient_agent import AbstractPolicyGradient
import torch


class REINFORCEAgent(AbstractPolicyGradient):
    """Implementation of REINFORCE algorithm.

    The returns and baseline targets are estimated with a Monte Carlo estimate.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

    """

    def _value_estimate(self, trajectories):
        values = []
        for trajectory in trajectories:
            val = torch.zeros_like(trajectory.reward)
            r = 0
            for i, reward in enumerate(reversed(trajectory.reward)):
                r = reward + self.gamma * r
                val[-1-i] = r

            values.append((val - val.mean()) / (val.std() + self.eps))

        return values

    def _td_base(self, state, action, reward, next_state, done, value_estimate=None):
        return self.baseline(state), value_estimate

    def _td_critic(self, state, action, reward, next_state, done):
        raise NotImplementedError
