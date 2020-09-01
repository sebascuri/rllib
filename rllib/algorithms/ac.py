"""Actor-Critic Algorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util import (
    discount_sum,
    get_entropy_and_logp,
    integrate,
    off_policy_weight,
    separated_kl,
    tensor_to_distribution,
)
from rllib.util.neural_networks import deep_copy_module, freeze_parameters
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
        super().__init__(*args, **kwargs)
        old_policy = deep_copy_module(self.policy)
        freeze_parameters(old_policy)
        self.old_policy = old_policy

        self.num_samples = num_samples
        self.standardize_returns = standardize_returns

        self.ope = ope

    def get_log_p_kl_entropy(self, state, action):
        """Get kl divergence and current policy at a given state.

        Compute the separated KL divergence between current and old policy.
        When the policy is a MultivariateNormal distribution, it compute the divergence
        that correspond to the mean and the covariance separately.

        When the policy is a Categorical distribution, it computes the divergence and
        assigns it to the mean component. The variance component is kept to zero.

        Parameters
        ----------
        state: torch.Tensor
            Empirical state distribution.

        action: torch.Tensor
            Actions sampled by pi_old.

        Returns
        -------
        log_p: torch.Tensor
            Log probability of actions according to current policy.
        log_p_old: torch.Tensor
            Log probability of actions according to old policy.
        kl_mean: torch.Tensor
            KL-Divergence due to the change in the mean between current and
            previous policy.
        kl_var: torch.Tensor
            KL-Divergence due to the change in the variance between current and
            previous policy.
        """
        pi = tensor_to_distribution(self.policy(state))
        pi_old = tensor_to_distribution(self.old_policy(state))

        entropy, log_p = get_entropy_and_logp(pi=pi, action=action)
        log_p_old = pi_old.log_prob(action)

        if isinstance(pi, torch.distributions.MultivariateNormal):
            kl_mean, kl_var = separated_kl(p=pi_old, q=pi)
        else:
            try:
                kl_mean = torch.distributions.kl_divergence(p=pi_old, q=pi).mean()
            except NotImplementedError:
                kl_mean = (log_p_old - log_p).mean()  # Approximate the KL with samples.
            kl_var = torch.zeros_like(kl_mean)

        num_t = self._info["num_trajectories"]
        self._info.update(
            kl_div=self._info["kl_div"] + (kl_mean + kl_var) / num_t,
            kl_mean=self._info["kl_mean"] + kl_mean / num_t,
            kl_var=self._info["kl_var"] + kl_var / num_t,
            entropy=self._info["entropy"] + entropy / num_t,
        )

        return log_p, log_p_old, kl_mean, kl_var, entropy

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        pi = tensor_to_distribution(self.policy(state))
        weight = off_policy_weight(
            pi.log_prob(action), trajectory.log_prob_action, full_trajectory=False
        )
        return weight * self.critic(state, action)

    def get_value_target(self, observation):
        """Get q function target."""
        if self.ope is not None:
            return self.ope(observation)
        if isinstance(self.critic_target, AbstractValueFunction):
            next_v = self.critic_target(observation.next_state)
            next_v = next_v * (1 - observation.done)
        elif isinstance(self.critic_target, AbstractQFunction):
            next_pi = tensor_to_distribution(self.policy(observation.next_state))
            next_v = integrate(
                lambda a: self.critic_target(observation.next_state, a),
                next_pi,
                num_samples=self.num_samples,
            )
            next_v = next_v * (1.0 - observation.done)
        else:
            raise RuntimeError(
                f"Critic Target type {type(self.critic_target)} not understood."
            )
        return self.reward_transformer(observation.reward) + self.gamma * next_v

    def actor_loss(self, observation):
        """Get Actor loss."""
        state, action, reward, next_state, done, *r = observation
        if self.policy.discrete_action:
            action = action.long()

        log_p, log_p_old, kl_mean, kl_var, entropy = self.get_log_p_kl_entropy(
            state, action
        )

        with torch.no_grad():
            returns = self.returns(observation)
            if self.standardize_returns:
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

        policy_loss = discount_sum(-log_p * returns, self.gamma)
        return Loss(
            policy_loss=policy_loss.mean(),
            regularization_loss=-self.entropy_regularization * entropy,
        )

    def reset_info(self, *args, **kwargs):
        """Reset AC info."""
        super().reset_info(*args, **kwargs)
        self._info.update(
            kl_div=torch.tensor(0.0),
            kl_mean=torch.tensor(0.0),
            kl_var=torch.tensor(0.0),
            entropy=torch.tensor(0.0),
        )
