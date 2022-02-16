"""Python Script Template."""

import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.multi_objective_reduction import MeanMultiObjectiveReduction
from rllib.util.neural_networks.utilities import DisableGradient
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

    def __init__(
        self,
        policy=None,
        critic=None,
        multi_objective_reduction=MeanMultiObjectiveReduction(dim=-1),
    ):
        super().__init__()
        self.policy = policy
        self.critic = critic
        self.multi_objective_reduction = multi_objective_reduction

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

        q = get_q_value_pathwise_gradients(
            critic=self.critic,
            state=state,
            action=action,
            multi_objective_reduction=self.multi_objective_reduction,
        )

        return Loss(policy_loss=-q)


def get_q_value_pathwise_gradients(critic, state, action, multi_objective_reduction):
    """Get Q-Value pathwise gradients.

    Parameters
    ----------
    critic: NNQFunction.
        Critic to backpropagate through.
    state: Tensor.
        State where to evaluate the q function.
    action: Tensor.
        Action where to evaluate the q function.
    multi_objective_reduction: Tensor.
        How to reduce the different outputs of the q function. (reward dimensions).
    """
    with DisableGradient(critic):
        q = critic(state, action)
        if isinstance(critic, NNEnsembleQFunction):
            q = q[..., 0]

    # Take multi-objective reduction.
    q = multi_objective_reduction(q)
    # Take mean over time coordinate.
    if q.dim() > 1:
        q = q.mean(dim=1)
    return q
