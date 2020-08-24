"""Trust Region Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay

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
        self, regularization=False, epsilon_mean=0.2, epsilon_var=1e-4, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if epsilon_var is None:
            epsilon_var = epsilon_mean

        if regularization:
            eta_mean = epsilon_mean
            eta_var = epsilon_var
            if not isinstance(eta_mean, ParameterDecay):
                eta_mean = Constant(eta_mean)
            if not isinstance(eta_var, ParameterDecay):
                eta_mean = Constant(eta_var)

            self.eta_mean = eta_mean
            self.eta_var = eta_var

            self.epsilon_mean = torch.tensor(0.0)
            self.epsilon_var = torch.tensor(0.0)

        else:  # Trust-Region: || KL(\pi_old || p) || < \epsilon
            self.eta_mean = Learnable(1.0, positive=True)
            self.eta_var = Learnable(1.0, positive=True)

            self.epsilon_mean = torch.tensor(epsilon_mean)
            self.epsilon_var = torch.tensor(epsilon_var)

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

        # Compute KL Loss.
        kl_loss = self.eta_mean().detach() * kl_mean + self.eta_var().detach() * kl_var

        # Compute Dual Loss.
        eta_mean_loss = self.eta_mean() * (self.epsilon_mean - kl_mean.detach())
        eta_var_loss = self.eta_var() * (self.epsilon_var - kl_var.detach())
        dual_loss = eta_mean_loss + eta_var_loss

        self._info.update(eta_mean=self.eta_mean(), eta_var=self.eta_var())

        return Loss(
            policy_loss=surrogate_loss, regularization_loss=kl_loss, dual_loss=dual_loss
        )
