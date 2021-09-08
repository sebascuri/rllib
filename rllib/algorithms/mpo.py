"""Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import repeat_along_dimension
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay
from rllib.util.utilities import tensor_to_distribution

from .abstract_algorithm import AbstractAlgorithm
from .policy_evaluation.retrace import ReTrace


class MPOLoss(nn.Module):
    """Maximum a Posterior Policy Optimization Losses.

    This method uses critic values under samples from a policy to construct a
    sample-based representation of the optimal policy. It then fits the parametric
    policy to this representation via supervised learning.

    Parameters
    ----------
    epsilon: float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy

    References
    ----------
    Abdolmaleki, et al. "Maximum a Posteriori Policy Optimisation." (2018). ICLR.
    """

    def __init__(self, epsilon=0.1, kl_regularization=False):
        super().__init__()

        if kl_regularization:
            eta = epsilon
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)

            self._eta = eta

            self.epsilon = torch.tensor(0.0)

        else:  # Trust-Region: || KL(q || q) || < \epsilon
            self._eta = Learnable(1.0, positive=True)

            self.epsilon = torch.tensor(epsilon)

    @property
    def eta(self):
        """Get MPO regularization parameter."""
        return self._eta().detach()

    def forward(self, q_values, action_log_p):
        """Return primal and dual loss terms from MPO.

        Parameters
        ----------
        q_values : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of values for
            state-action pairs.
        action_log_p : torch.Tensor
            A [n_action_samples, state_batch, 1] tensor of log probabilities
            of the corresponding actions under the policy.
        """
        # Make sure the lagrange multipliers stay positive.
        # self.project_etas()

        # E-step: Solve Problem (7).
        # Create a weighed, sample-based representation of the optimal policy q Eq(8).
        # Compute the dual loss for the constraint KL(q || old_pi) < eps.
        q_values = q_values.detach() * (torch.tensor(1.0) / self._eta())
        normalizer = torch.logsumexp(q_values, dim=0)
        num_actions = torch.tensor(1.0 * action_log_p.shape[0])

        dual_loss = self._eta() * (
            self.epsilon + torch.mean(normalizer) - torch.log(num_actions)
        )
        # non-parametric representation of the optimal policy.
        weights = torch.exp(q_values - normalizer).detach()

        # M-step: # E-step: Solve Problem (10).
        # Fit the parametric policy to the representation form the E-step.
        # Maximize the log_likelihood of the weighted log probabilities, subject to the
        # KL divergence between the old_pi and the new_pi to be smaller than epsilon.

        weighted_log_p = torch.sum(weights * action_log_p, dim=0)
        log_likelihood = weighted_log_p

        return Loss(policy_loss=-log_likelihood.mean(), dual_loss=dual_loss)


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
    num_samples: int.
        Number of action samples to approximate integral.
    gamma: float
        The discount factor.

    References
    ----------
    Abdolmaleki, et al. (2018)
    Maximum a Posteriori Policy Optimisation. ICLR.

    """

    def __init__(
        self, num_samples=15, epsilon=0.1, kl_regularization=False, *args, **kwargs
    ):
        super().__init__(
            num_samples=num_samples,
            kl_regularization=kl_regularization,
            *args,
            **kwargs,
        )
        self.mpo_loss = MPOLoss(epsilon, kl_regularization)

    def post_init(self):
        """Call after initialization to initialize other modules."""
        super().post_init()
        self.ope = ReTrace(
            policy=self.old_policy,
            critic=self.critic_target,
            gamma=self.gamma,
            num_samples=self.num_samples,
            lambda_=1.0,
        )

    def get_value_target(self, observation):
        """Get value target."""
        if self.ope is not None:
            return self.ope(observation)
        next_v = self.value_target(observation.next_state)
        next_v = next_v * (1.0 - observation.done)
        return self.get_reward(observation) + self.gamma * next_v

    def compute_mpo_loss(self, state, action):
        """Compute mpo loss for a given set of state/action pairs."""
        pi_dist = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        log_p = pi_dist.log_prob(action)

        q_values = self.critic_target(state, action)

        mpo_loss = self.mpo_loss(q_values=q_values, action_log_p=log_p).reduce(
            self.criterion.reduction
        )
        self._info.update(mpo_eta=self.mpo_loss.eta)
        return mpo_loss

    def actor_loss(self, observation):
        """Compute actor loss."""
        state = repeat_along_dimension(
            observation.state, number=self.num_samples, dim=0
        )
        pi = tensor_to_distribution(self.old_policy(state), **self.policy.dist_params)
        action = self.policy.action_scale * pi.sample().clamp(-1.0, 1.0)
        return self.compute_mpo_loss(state, action)
