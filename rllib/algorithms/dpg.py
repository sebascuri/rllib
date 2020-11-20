"""Deterministic Policy Gradient Algorithm."""

from rllib.policy.utilities import DistParams

from .abstract_algorithm import AbstractAlgorithm


class DPG(AbstractAlgorithm):
    r"""Implementation of DPG algorithm.

    DPG is an off-policy model-free control algorithm.

    The DPG algorithm is an actor-critic algorithm that has a policy that estimates:
    .. math:: a = \pi(s) = \argmax_a Q(s, a)


    Parameters
    ----------
    critic: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Silver, David, et al. (2014)
    Deterministic policy gradient algorithms. JMLR.

    Lillicrap et. al. (2016).
    Continuous Control with Deep Reinforcement Learning. ICLR.
    """

    def __init__(self, policy_noise=0.0, noise_clip=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_target.dist_params.update(
            add_noise=True, policy_noise=policy_noise, noise_clip=noise_clip
        )
        self.value_function.policy = self.policy
        self.value_target.policy = self.policy_target

    def set_policy(self, new_policy):
        """Set new policy."""
        policy_target_dist_params = self.policy_target.dist_params
        super().set_policy(new_policy)
        self.policy_target.dist_params = policy_target_dist_params
        self.value_function.policy = self.policy
        self.value_target.policy = self.policy_target

    def actor_loss(self, observation):
        """Get Actor Loss."""
        with DistParams(self.policy, add_noise=False):
            return self.pathwise_loss(observation).reduce(self.criterion.reduction)
