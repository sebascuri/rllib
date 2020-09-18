"""Implementation of Advantage-Actor Critic Agent."""
import torch.nn.modules.loss as loss

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

    def __init__(self, policy, critic, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)

        self.algorithm = A2C(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
