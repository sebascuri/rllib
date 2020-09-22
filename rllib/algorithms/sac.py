"""Soft Actor-Critic Algorithm."""

from .abstract_algorithm import AbstractAlgorithm


class SAC(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    Stochastic Actor. ICML.

    Haarnoja, T., Zhou, A., ... & Levine, S. (2018).
    Soft actor-critic algorithms and applications. arXiv.
    """

    def __init__(self, eta=0.2, entropy_regularization=False, *args, **kwargs):
        super().__init__(
            eta=eta, entropy_regularization=entropy_regularization, *args, **kwargs
        )
        assert (
            len(self.policy.dim_action) == 1
        ), "Only Nx1 continuous actions implemented."

    def post_init(self):
        """Set derived modules after initialization."""
        super().post_init()
        self.policy.dist_params.update(tanh=True)
        self.policy_target.dist_params.update(tanh=True)

    def actor_loss(self, observation):
        """Get Actor Loss."""
        return self.pathwise_loss(observation).reduce(self.criterion.reduction)
