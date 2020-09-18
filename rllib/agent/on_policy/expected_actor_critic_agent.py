"""Implementation of Expected-Actor Critic Agent."""
import torch.nn.modules.loss as loss

from rllib.algorithms.eac import ExpectedActorCritic

from .actor_critic_agent import ActorCriticAgent


class ExpectedActorCriticAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Ciosek, K., & Whiteson, S. (2018).
    Expected policy gradients. AAAI.
    """

    def __init__(self, policy, critic, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)
        self.algorithm = ExpectedActorCritic(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
