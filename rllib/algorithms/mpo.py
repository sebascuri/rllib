"""Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    repeat_along_dimension,
)
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import (
    get_entropy_and_log_p,
    separated_kl,
    tensor_to_distribution,
)

from .abstract_algorithm import AbstractAlgorithm
from .policy_evaluation.retrace import ReTrace


class MPOWorker(nn.Module):
    """Maximum a Posterior Policy Optimization Losses.

    This method uses critic values under samples from a policy to construct a
    sample-based representation of the optimal policy. It then fits the parametric
    policy to this representation via supervised learning.

    Parameters
    ----------
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean: float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.
    epsilon_var: float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.

    References
    ----------
    Abdolmaleki, et al. "Maximum a Posteriori Policy Optimisation." (2018). ICLR.
    """

    def __init__(
        self, epsilon=0.1, epsilon_mean=0.1, epsilon_var=0.0001, regularization=False
    ):
        super().__init__()
        if epsilon_var is None:
            epsilon_var = 0.0

        if regularization:
            eta = epsilon
            eta_mean = epsilon_mean
            eta_var = epsilon_var
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            if not isinstance(eta_mean, ParameterDecay):
                eta_mean = Constant(eta_mean)
            if not isinstance(eta_var, ParameterDecay):
                eta_var = Constant(eta_var)

            self.eta = eta
            self.eta_mean = eta_mean
            self.eta_var = eta_var

            self.epsilon = torch.tensor(0.0)
            self.epsilon_mean = torch.tensor(0.0)
            self.epsilon_var = torch.tensor(0.0)

        else:  # Trust-Region: || KL(q || \pi_old) || < \epsilon
            self.eta = Learnable(1.0, positive=True)
            self.eta_mean = Learnable(1.0, positive=True)
            self.eta_var = Learnable(1.0, positive=True)

            self.epsilon = torch.tensor(epsilon)
            self.epsilon_mean = torch.tensor(epsilon_mean)
            self.epsilon_var = torch.tensor(epsilon_var)

    def forward(self, q_values, action_log_p, kl_mean, kl_var):
        """Return primal and dual loss terms from MMPO.

        Parameters
        ----------
        q_values : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of values for
            state-action pairs.
        action_log_p : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of log probabilities
            of the corresponding actions under the policy.
        kl_mean : torch.Tensor
            A float corresponding to the KL divergence.
        kl_var : torch.Tensor
            A float corresponding to the KL divergence.
        """
        # Make sure the lagrange multipliers stay positive.
        # self.project_etas()

        # E-step: Solve Problem (7).
        # Create a weighed, sample-based representation of the optimal policy q Eq(8).
        # Compute the dual loss for the constraint KL(q || old_pi) < eps.
        q_values = q_values.detach() * (torch.tensor(1.0) / self.eta())
        normalizer = torch.logsumexp(q_values, dim=0)
        num_actions = torch.tensor(1.0 * action_log_p.shape[0])

        dual_loss = self.eta() * (
            self.epsilon + torch.mean(normalizer) - torch.log(num_actions)
        )
        # non-parametric representation of the optimal policy.
        weights = torch.exp(q_values - normalizer.detach())

        # M-step: # E-step: Solve Problem (10).
        # Fit the parametric policy to the representation form the E-step.
        # Maximize the log_likelihood of the weighted log probabilities, subject to the
        # KL divergence between the old_pi and the new_pi to be smaller than epsilon.

        weighted_log_p = torch.sum(weights * action_log_p, dim=0)
        log_likelihood = torch.mean(weighted_log_p)

        kl_loss = self.eta_mean().detach() * kl_mean + self.eta_var().detach() * kl_var

        eta_mean_loss = self.eta_mean() * (self.epsilon_mean - kl_mean.detach())
        eta_var_loss = self.eta_var() * (self.epsilon_var - kl_var.detach())

        dual_loss = dual_loss + eta_mean_loss + eta_var_loss

        return Loss(
            policy_loss=-log_likelihood,
            regularization_loss=kl_loss,
            dual_loss=dual_loss,
        )


class MPO(AbstractAlgorithm):
    """Maximum a Posteriori Policy Optimizaiton.

    The MPO algorithm returns a loss that is a combination of three losses.

    - The dual loss associated with the variational distribution (Eq. 9)
    - The dual loss associated with the KL-hard constraint (Eq. 12).
    - The primal loss associated with the policy fitting term (Eq. 12).
    - A policy evaluation loss (Eq. 13).

    To compute the primal and dual losses, it uses the MPOLoss module.

    Parameters
    ----------
    policy : AbstractPolicy
    q_function : AbstractQFunction
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean: float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    epsilon_var: float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    num_action_samples: int.
        Number of action samples to approximate integral.
    gamma: float
        The discount factor.

    References
    ----------
    Abdolmaleki, et al. (2018)
    Maximum a Posteriori Policy Optimisation. ICLR.

    """

    def __init__(
        self,
        num_action_samples=15,
        epsilon=0.1,
        epsilon_mean=0.0,
        epsilon_var=0.0001,
        regularization=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        old_policy = deep_copy_module(self.policy)
        freeze_parameters(old_policy)

        self.old_policy = old_policy
        self.critic_target = deep_copy_module(self.critic)
        self.num_action_samples = num_action_samples

        self.mpo_loss = MPOWorker(epsilon, epsilon_mean, epsilon_var, regularization)
        self.ope = ReTrace(
            policy=self.old_policy,
            critic=self.critic_target,
            gamma=self.gamma,
            num_samples=self.num_action_samples,
            lambda_=1.0,
        )

    def get_kl_and_pi(self, state):
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

        Returns
        -------
        kl_mean: torch.Tensor
            KL-Divergence due to the change in the mean between current and
            previous policy.

        kl_var: torch.Tensor
            KL-Divergence due to the change in the variance between current and
            previous policy.

        pi_dist: torch.distribution.Distribution
            Current policy distribution.
        """
        pi_dist = tensor_to_distribution(self.policy(state))
        pi_dist_old = tensor_to_distribution(self.old_policy(state))

        if isinstance(pi_dist, torch.distributions.MultivariateNormal):
            kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)
        else:
            kl_mean = torch.distributions.kl_divergence(p=pi_dist_old, q=pi_dist).mean()
            kl_var = torch.zeros_like(kl_mean)

        num_t = self._info["num_trajectories"]
        self._info.update(
            kl_div=self._info["kl_div"] + (kl_mean + kl_var) / num_t,
            kl_mean=self._info["kl_mean"] + kl_mean / num_t,
            kl_var=self._info["kl_var"] + kl_var / num_t,
            entropy=self._info["entropy"] + pi_dist.entropy().mean() / num_t,
        )

        return kl_mean, kl_var, pi_dist

    def get_value_target(self, observation):
        """Get value target."""
        if self.ope is not None:
            return self.ope(observation)
        next_pi = tensor_to_distribution(self.old_policy(observation.next_state))
        next_action = next_pi.sample()

        next_values = self.critic_target(observation.next_state, next_action)
        next_values = next_values * (1.0 - observation.done)

        reward = self.reward_transformer(observation.reward)
        value_target = reward + self.gamma * next_values
        return value_target

    def actor_loss(self, observation):
        """Compute actor loss."""
        state, action, reward, next_state, done, *r = observation

        state = repeat_along_dimension(state, number=self.num_action_samples, dim=0)

        kl_mean, kl_var, pi_dist = self.get_kl_and_pi(state)

        sampled_action = pi_dist.sample()
        entropy, log_p = get_entropy_and_log_p(pi_dist, sampled_action, 1)

        mpo_loss = self.mpo_loss(
            q_values=self.critic_target(state, sampled_action),
            action_log_p=log_p,
            kl_mean=kl_mean,
            kl_var=kl_var,
        )

        self._info.update(
            eta=self.mpo_loss.eta(),
            eta_mean=self.mpo_loss.eta_mean(),
            eta_var=self.mpo_loss.eta_var(),
        )

        return mpo_loss

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def reset_info(self, *args, **kwargs):
        """Reset MPO info."""
        super().reset_info(*args, **kwargs)
        self._info.update(
            kl_div=torch.tensor(0.0),
            kl_mean=torch.tensor(0.0),
            kl_var=torch.tensor(0.0),
            entropy=torch.tensor(0.0),
        )
