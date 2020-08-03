"""Implementation of Advantage-Actor Critic Agent."""

from rllib.algorithms.a2c import A2C

from .actor_critic_agent import ActorCriticAgent


class A2CAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Mnih, V., et al. (2016).
    Asynchronous methods for deep reinforcement learning. ICML.
    """

    def __init__(self, policy, critic, optimizer, criterion, *args, **kwargs):
        super().__init__(
            policy=policy,
            optimizer=optimizer,
            critic=critic,
            criterion=criterion,
            *args,
            **kwargs,
        )

        self.algorithm = A2C(policy, critic, criterion(reduction="none"), self.gamma)
        self.policy = self.algorithm.policy
