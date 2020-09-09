"""On-Policy Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions

from rllib.util.utilities import get_entropy_and_log_p

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

    def post_init(self) -> None:
        """Set derived modules after initialization."""
        super().post_init()
        self.ope = VTrace(
            policy=self.policy,
            critic=self.critic_target,
            num_samples=self.num_samples,
            rho_bar=1.0,
            gamma=self.gamma,
            lambda_=1.0,
        )

    def actor_loss(self, observation):
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

        # Since actions are on-policy, advantage is correct but
        # we should use IS in the off-policy case.
        advantage = value_target - value_prediction
        _, log_p = get_entropy_and_log_p(pi_dist, action, self.policy.action_scale)

        k = int(self.top_k_fraction * state.shape[0])
        _, idx_top_k = torch.topk(advantage.mean(-1), k=k, dim=0, largest=True)
        advantage_top_k = advantage[idx_top_k]
        log_p_top_k = log_p[idx_top_k]

        mpo_loss = self.mpo_loss(
            q_values=advantage_top_k.unsqueeze(0),  # 1-st dim is MPO action samples.
            action_log_p=log_p_top_k.unsqueeze(0),
            kl_mean=kl_mean,
            kl_var=kl_var,
        )

        self._info.update(
            eta=self.mpo_loss.eta(),
            eta_mean=self.mpo_loss.eta_mean(),
            eta_var=self.mpo_loss.eta_var(),
        )

        return mpo_loss
