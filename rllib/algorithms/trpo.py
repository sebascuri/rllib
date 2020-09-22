"""Trust Region Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.util.value_estimation import discount_cumsum

from .gaac import GAAC


class TRPO(GAAC):
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
        self,
        epsilon_mean=0.01,
        epsilon_var=None,
        kl_regularization=False,
        lambda_=0.95,
        monte_carlo_target=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            kl_regularization=kl_regularization,
            lambda_=lambda_,
            *args,
            **kwargs,
        )
        self.monte_carlo_target = monte_carlo_target

    def get_value_target(self, observation):
        """Get the Q-Target."""
        if self.ope is not None:
            return self.ope(observation)
        elif self.monte_carlo_target:  # Compute as \sum_returns.
            final_state = observation.next_state[-1:]
            reward_to_go = self.critic_target(final_state) * (
                1.0 - observation.done[-1:]
            )
            value_target = discount_cumsum(
                torch.cat((observation.reward, reward_to_go)), gamma=self.gamma
            )[:-1]
            return (value_target - value_target.mean()) / (
                value_target.std() + self.eps
            )
        else:  # Compute as ADV(S, A) + V(S).
            adv = self.returns(observation)
            if self.standardize_returns:
                adv = (adv - adv.mean()) / (adv.std() + self.eps)

            return adv + self.critic_target(observation.state)

    def actor_loss(self, observation):
        """Get actor loss."""
        return self.score_actor_loss(observation, linearized=True).reduce(
            self.criterion.reduction
        )
