"""On-Policy Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions

from .abstract_algorithm import Loss
from .mpo import MPO
from .policy_evaluation.vtrace import VTrace


class VMPO(MPO):
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
    """

    def __init__(self, top_k_fraction=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.top_k_fraction = top_k_fraction
        if not (0 <= top_k_fraction <= 1):
            raise ValueError(
                f"Top-k fraction should be in [0, 1]. Got {top_k_fraction} instead."
            )

        self.trace = VTrace(
            policy=self.policy,
            critic=self.critic_target,
            num_samples=self.num_action_samples,
            rho_bar=1.0,
            gamma=self.gamma,
            lambda_=1.0,
        )

    def forward_slow(self, observation):
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
        state, action, *r = observation

        kl_mean, kl_var, pi_dist = self.get_kl_and_pi(state)
        value_prediction = self.critic(state)

        with torch.no_grad():
            value_target = self.get_value_target(observation)

        advantage = value_target - value_prediction
        action_log_probs = pi_dist.log_prob(action / self.policy.action_scale)

        k = int(self.top_k_fraction * state.shape[0])
        advantage_top_k, idx_top_k = torch.topk(advantage, k=k, dim=0, largest=True)
        action_log_probs_top_k = action_log_probs[idx_top_k.squeeze()]

        # Since actions come from policy, value is the expected q-value but we should
        # correct inf the off-policy case.
        losses = self.mpo_loss(
            q_values=advantage_top_k,
            action_log_probs=action_log_probs_top_k,
            kl_mean=kl_mean,
            kl_var=kl_var,
        )

        critic_loss = self.critic_loss(observation)

        dual_loss = losses.dual_loss.mean()
        policy_loss = losses.policy_loss.mean()
        combined_loss = critic_loss.critic_loss + dual_loss + policy_loss

        self._info = {
            "kl_div": kl_mean + kl_var,
            "kl_mean": kl_mean,
            "kl_var": kl_var,
            "eta": self.mpo_loss.eta(),
            "eta_mean": self.mpo_loss.eta_mean(),
            "eta_var": self.mpo_loss.eta_var(),
        }

        return Loss(
            loss=combined_loss,
            dual_loss=dual_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss.critic_loss,
            td_error=critic_loss.td_error,
        )
