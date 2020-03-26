"""Implementation of GP-UCB algorithm."""

import gpytorch
import torch

from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy


class GPUCBPolicy(AbstractPolicy):
    """GP UCB Policy.

    Implementation of GP-UCB algorithm.
    GP-UCB uses a GP to maintain the predictions of a distribution over actions.

    The algorithm selects the action as:
    x = arg max mean(x) + beta * std(x)
    where mean(x) and std(x) are the mean and standard devations of the GP at loc x.

    Parameters
    ----------
    gp: initialized GP model.
    x: discretization of domain.
    beta: exploration parameter.

    References
    ----------
    Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2009).
    Gaussian process optimization in the bandit setting: No regret and experimental
    design.
    """

    def __init__(self, gp, x, beta=2.0):
        super().__init__(dim_state=1, dim_action=x.shape[0],
                         num_states=1, num_actions=-1, deterministic=True)
        self.gp = gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.x = x
        self.beta = beta

    def forward(self, state):
        """Call the GP-UCB algorithm."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.gp(self.x)
            ucb = pred.mean + self.beta * pred.stddev

            max_id = torch.argmax(ucb)
            next_point = self.x[[[max_id]]]
            return next_point, torch.zeros(1)


class GPUCBAgent(AbstractAgent):
    """Agent that implements the GP-UCB algorithm.

    Parameters
    ----------
    gp: initialized GP model.
    x: discretization of domain.
    beta: exploration parameter.

    References
    ----------
    Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2009).
    Gaussian process optimization in the bandit setting: No regret and experimental
    design.
    """

    def __init__(self, environment, gp, x, beta=2.0):
        self.policy = GPUCBPolicy(gp, x, beta)
        super().__init__(environment, gamma=1, exploration_episodes=0,
                         exploration_steps=0)

    def observe(self, observation) -> None:
        """Observe and update GP."""
        super().observe(observation)
        self.policy.gp = self.policy.gp.get_fantasy_model(observation.action,
                                                          observation.reward)
