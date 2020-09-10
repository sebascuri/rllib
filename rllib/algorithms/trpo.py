"""Trust Region Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.dataset.datatypes import Loss

from .kl_loss import KLLoss
from .ppo import PPO


class TRPO(PPO):
    """Trust Region Policy Optimization algorithm..

    The TRPO algorithm returns a loss that is a combination of three losses.

    - The surrogate objective.
    - The value function error.
    - The dual loss that arises from the violation of the KL constraint.

    Parameters
    ----------
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    regularization: bool.
        Flag to indicate whether to regularize or do a trust region.
    epsilon_mean: ParameterDecay, float, optional.
        Kl Divergence of the Mean.
    epsilon_var: ParameterDecay, float, optional.
        Kl Divergence of the variance.
    lambda_: float, optional. (default=0.97).
        Parameter for Advantage estimation.
    gamma: float, optional. (default=1).
        Discount factor.

    References
    ----------
    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    Trust region policy optimization. ICML.
    """

    def __init__(
        self, epsilon_mean=0.2, epsilon_var=1e-4, regularization=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kl_loss = KLLoss(epsilon_mean, epsilon_var, regularization)

    def actor_loss(self, trajectory):
        """Get actor loss."""
        state, action, reward, next_state, done, *r = trajectory
        log_p, log_p_old, kl_mean, kl_var, entropy = self.get_log_p_kl_entropy(
            state, action
        )
        with torch.no_grad():
            adv = self.returns(trajectory)
            if self.standardize_returns:
                adv = (adv - adv.mean()) / (adv.std() + self.eps)

        # Compute surrogate loss.
        ratio = torch.exp(log_p - log_p_old)
        surrogate_loss = -(ratio * adv).mean()

        # Compute Policy Loss
        actor_loss = Loss(
            policy_loss=surrogate_loss,
            regularization_loss=-self.entropy_regularization * entropy,
        )

        # Compute KL Loss.
        kl_loss = self.kl_loss(kl_mean, kl_var)
        self._info.update(
            eta_mean=self.kl_loss.eta_mean(), eta_var=self.kl_loss.eta_var()
        )

        return actor_loss + kl_loss
