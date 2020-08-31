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

    def __init__(self, policy, critic, *args, **kwargs):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)

        self.algorithm = A2C(
            policy=policy,
            critic=critic,
            criterion=self.algorithm.criterion,
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy
