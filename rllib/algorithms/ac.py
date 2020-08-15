"""Actor-Critic Algorithm."""
import torch

from rllib.util import (
    discount_sum,
    get_entropy_and_logp,
    integrate,
    separated_kl,
    tensor_to_distribution,
)
from rllib.util.neural_networks import deep_copy_module, freeze_parameters
from rllib.value_function import AbstractQFunction, AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, Loss


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
    Policy gradient methods for reinforcement learning with function approximation.NIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NIPS.
    """

    def __init__(
        self,
        num_samples=15,
        entropy_regularization=0.0,
        standarize_returns=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        old_policy = deep_copy_module(self.policy)
        freeze_parameters(old_policy)
        self.old_policy = old_policy

        self.num_samples = num_samples
        self.entropy_regularization = entropy_regularization
        self.standardize_returns = standarize_returns

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

        return log_p, log_p_old, kl_mean, kl_var, entropy

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def returns(self, trajectory):
        """Estimate the returns of a trajectory."""
        state, action = trajectory.state, trajectory.action
        return self.critic(state, action)

    def get_value_target(self, observation):
        """Get q function target."""
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

        actor_loss = discount_sum(-log_p * returns, self.gamma)
        return Loss(
            loss=actor_loss, policy_loss=actor_loss, regularization_loss=-entropy
        )

    def forward_slow(self, trajectories):
        """Compute the losses iterating through the trajectories."""
        policy_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        td_error = torch.tensor(0.0)
        self._info.update(kl_div=torch.tensor(0.0))

        for trajectory in trajectories:
            # ACTOR LOSS
            actor_loss = self.actor_loss(trajectory)
            policy_loss += actor_loss.policy_loss.mean()
            entropy_loss += actor_loss.regularization_loss.mean()

            # CRITIC LOSS
            critic_loss_ = self.critic_loss(trajectory)
            critic_loss += critic_loss_.critic_loss.mean()
            td_error += critic_loss_.td_error.mean()

        num_trajectories = len(trajectories)
        self._info.update(kl_div=self._info["kl_div"] / num_trajectories)
        loss = policy_loss + critic_loss + self.entropy_regularization * entropy_loss
        return Loss(
            loss=loss / num_trajectories,
            policy_loss=policy_loss / num_trajectories,
            critic_loss=critic_loss / num_trajectories,
            regularization_loss=entropy_loss / num_trajectories,
            td_error=td_error / num_trajectories,
        )
