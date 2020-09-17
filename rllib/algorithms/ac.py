"""Actor-Critic Algorithm."""
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

    def __init__(self, num_samples=15, standardize_returns=True, *args, **kwargs):
        super().__init__(num_samples=num_samples, *args, **kwargs)
        self.standardize_returns = standardize_returns

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        weight = self.get_ope_weight(state, action, trajectory.log_prob_action)
        return weight * self.critic(state, action)

    def actor_loss(self, observation):
        """Get Actor loss."""
        return self.score_actor_loss(observation, linearized=False).reduce(
            self.criterion.reduction
        )
