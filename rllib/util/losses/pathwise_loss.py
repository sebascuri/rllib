"""Python Script Template."""

import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction


class PathwiseLoss(nn.Module):
    """Compute pathwise loss.

    References
    ----------
    Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020).
    Monte Carlo Gradient Estimation in Machine Learning. JMLR.

    Parmas, P., Rasmussen, C. E., Peters, J., & Doya, K. (2018).
    PIPPS: Flexible model-based policy search robust to the curse of chaos. ICML.

    Silver, David, et al. (2014)
    Deterministic policy gradient algorithms. JMLR.

    O'Donoghue, B., Munos, R., Kavukcuoglu, K., & Mnih, V. (2017)
    Combining policy gradient and Q-learning. ICLR.

    Gu, S. S., et al. (2017)
    Interpolated policy gradient: Merging on-policy and off-policy gradient estimation
    for deep reinforcement learning. NeuRIPS.

    Wang, Z., et al. (2017)
    Sample efficient actor-critic with experience replay. ICRL.
    """

    def __init__(self, policy=None, critic=None):
        super().__init__()
        self.policy = policy
        self.critic = critic

    def set_policy(self, new_policy):
        """Set policy."""
        self.policy = new_policy
        try:
            self.critic.set_policy(new_policy)
        except AttributeError:
            pass

    def forward(self, observation):
        """Compute path-wise loss."""
        if self.policy is None or self.critic is None:
            return Loss()
        state = observation.state
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        action = self.policy.action_scale * pi.rsample().clamp(-1, 1)

        with DisableGradient(self.critic):
            q = self.critic(state, action)
            if isinstance(self.critic, NNEnsembleQFunction):
                q = q[..., 0]

        # Take mean over time coordinate.
        if q.dim() < 1:
            q = q.mean(dim=1)

        return Loss(policy_loss=-q)
