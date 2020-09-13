"""MPC Algorithms."""

import torch

from rllib.util.value_estimation import mb_return

from .random_shooting import RandomShooting


class PolicyShooting(RandomShooting):
    r"""Policy shooting the MPC problem by sampling from a policy.

    Parameters
    ----------
    policy: AbstractPolicy.

    Other Parameters
    ----------------
    See Also: RandomShooting.

    References
    ----------
    Hong, Z. W., Pajarinen, J., & Peters, J. (2019).
    Model-based lookahead reinforcement learning. arXiv.
    """

    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def forward(self, state, **kwargs):
        """Get best action."""
        self.dynamical_model.eval()

        value, trajectory = mb_return(
            state,
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=self.policy,
            num_steps=self.horizon,
            gamma=self.gamma,
            num_samples=self.num_samples,
            value_function=self.terminal_reward,
            termination_model=self.termination_model,
        )
        actions = trajectory.action
        idx = torch.topk(value, k=self.num_elites, largest=True)[1]

        # Return first action and the mean over the elite samples.
        return actions[0, idx].mean(0)
