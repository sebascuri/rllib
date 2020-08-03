"""On-Policy Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    update_parameters,
)
from rllib.util.utilities import RewardTransformer, separated_kl, tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm, MPOLoss
from .mpo import MPOWorker


class VMPO(AbstractAlgorithm):
    """On-Policy Maximum a Posteriori Policy Optimizaiton.

    The V-MPO algorithm returns a loss that is a combination of four losses.

    - The dual loss associated with the variational distribution (Eq. 4)
    - The dual loss associated with the KL-hard constraint (Eq. 5).
    - The primal loss associated with the policy fitting term (Eq. 3).
    - A policy evaluation loss (Eq. 6).

    To compute the primal and dual losses, it uses the MPOLoss module.

    Parameters
    ----------
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean: float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    epsilon_var: float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    gamma: float
        The discount factor.

    References
    ----------
    Song, H. F., et al. (2019).
    V-MPO: on-policy maximum a posteriori policy optimization for discrete and
    continuous control. ICLR.

    TODO: Add VTrace for policy evaluation.
    """

    def __init__(
        self,
        policy,
        value_function,
        criterion,
        epsilon=0.1,
        epsilon_mean=0.0,
        epsilon_var=0.0001,
        regularization=False,
        top_k_fraction=0.5,
        gamma=0.99,
        reward_transformer=RewardTransformer(),
    ):
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__()
        self.old_policy = old_policy
        self.policy = policy
        self.value_function = value_function
        self.value_target = deep_copy_module(value_function)

        self.gamma = gamma
        self.reward_transformer = reward_transformer

        self.top_k_fraction = top_k_fraction
        if not (0 <= top_k_fraction <= 1):
            raise ValueError(
                f"Top-k fraction should be in [0, 1]. Got {top_k_fraction} instead."
            )

        self.mpo_loss = MPOWorker(epsilon, epsilon_mean, epsilon_var, regularization)
        self.value_loss = criterion(reduction="mean")

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

        return kl_mean, kl_var, pi_dist

    def get_value_target(self, reward, next_state, done):
        """Get the value function target."""
        next_v = self.value_target(next_state) * (1.0 - done)
        v_target = self.reward_transformer(reward) + self.gamma * next_v
        return v_target

    def forward(self, observation):
        """Compute the losses for one step of MPO.

        Parameters
        ----------
        observation: Observation
            Observation batch.

        Returns
        -------
        loss: torch.Tensor
            The combined loss
        value_loss: torch.Tensor
            The loss of the value function approximation.
        policy_loss: torch.Tensor
            The kl-regularized fitting loss for the policy.
        eta_loss: torch.Tensor
            The loss for the lagrange multipliers.
        kl_div: torch.Tensor
            The average KL divergence of the policy.
        """
        state, action, reward, next_state, done, *r = observation

        value_prediction = self.value_function(state)

        kl_mean, kl_var, pi_dist = self.get_kl_and_pi(state)

        with torch.no_grad():
            value_target = self.get_value_target(reward, next_state, done)

        advantage = value_target - value_prediction
        action_log_probs = pi_dist.log_prob(action / self.policy.action_scale)

        k = int(self.top_k_fraction * state.shape[0])
        advantage_top_k, idx_top_k = torch.topk(advantage, k=k, dim=0, largest=True)
        action_log_probs_top_k = action_log_probs[idx_top_k.squeeze()]

        # Since actions come from policy, value is the expected q-value
        losses = self.mpo_loss(
            q_values=advantage_top_k,
            action_log_probs=action_log_probs_top_k,
            kl_mean=kl_mean,
            kl_var=kl_var,
        )

        value_loss = self.value_loss(value_prediction, value_target)
        td_error = value_prediction - value_target

        dual_loss = losses.dual_loss.mean()
        policy_loss = losses.primal_loss.mean()
        combined_loss = value_loss + dual_loss + policy_loss

        self._info = {
            "kl_div": kl_mean + kl_var,
            "kl_mean": kl_mean,
            "kl_var": kl_var,
            "eta": self.mpo_loss.eta(),
            "eta_mean": self.mpo_loss.eta_mean(),
            "eta_var": self.mpo_loss.eta_var(),
        }

        return MPOLoss(
            loss=combined_loss,
            dual=dual_loss,
            policy_loss=policy_loss,
            critic_loss=value_loss,
            td_error=td_error,
        )

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self):
        """Update target networks."""
        update_parameters(
            self.value_target, self.value_function, tau=self.value_function.tau
        )
