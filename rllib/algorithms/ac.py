"""Actor-Critic Algorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.value_estimation import discount_sum
from rllib.value_function import AbstractQFunction, AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm


class ActorCritic(AbstractAlgorithm):
    r"""Implementation of Policy Gradient algorithm.

    Policy-Gradient is an on-policy model-free control algorithm.
    Policy-Gradient computes the policy gradient using a critic to estimate the returns
    (sum of discounted rewards).

    The Policy-Gradient algorithm is a policy gradient algorithm that estimates the
    gradient:
    .. math:: \grad J = \int_{\tau} \grad \log \pi(s_t) Q(s_t, a_t),
    where the previous integral is computed through samples (s_t, a_t) samples.


    Parameters
    ----------
    policy: AbstractPolicy
        Policy to optimize.
    critic: AbstractQFunction
        Critic that evaluates the current policy.
    criterion: _Loss
        Criterion to optimize the baseline.
    gamma: float
        Discount factor.

    References
    ----------
    Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).
    Policy gradient methods for reinforcement learning with function approximation.
    NeurIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NeurIPS.

    Degris, T., White, M., & Sutton, R. S. (2012).
    Off-policy actor-critic. ICML
    """

    def __init__(
        self, num_samples=15, standardize_returns=True, ope=None, *args, **kwargs
    ):
        super().__init__(num_samples=num_samples, *args, **kwargs)
        self.standardize_returns = standardize_returns
        self.ope = ope

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        weight = self.get_ope_weight(state, action, trajectory.log_prob_action)
        return weight * self.critic(state, action)

    def get_value_target(self, observation):
        """Get q function target."""
        if self.ope is not None:
            return self.ope(observation)
        if isinstance(self.critic_target, AbstractValueFunction):
            next_v = self.critic_target(observation.next_state)
        elif isinstance(self.critic_target, AbstractQFunction):
            next_v = self.value_target(observation.next_state)
        else:
            raise RuntimeError(
                f"Critic Target type {type(self.critic_target)} not understood."
            )
        next_v = next_v * (1.0 - observation.done)
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def actor_loss(self, observation):
        """Get Actor loss."""
        state, action, reward, next_state, done, *r = observation

        log_p, _, _, _, _ = self.get_log_p_kl_entropy(state, action)

        with torch.no_grad():
            returns = self.returns(observation)
            if self.standardize_returns:
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

        policy_loss = discount_sum(-log_p * returns, self.gamma)
        return Loss(policy_loss=policy_loss).reduce(self.criterion.reduction)
