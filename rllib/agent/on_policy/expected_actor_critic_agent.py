"""Implementation of Expected-Actor Critic Agent."""
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

    def __init__(self, *args, **kwargs):
        super().__init__(algorithm_=ExpectedActorCritic, *args, **kwargs)
