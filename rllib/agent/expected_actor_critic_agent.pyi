from rllib.algorithms.eac import ExpectedActorCritic
from .actor_critic_agent import ActorCriticAgent


class ExpectedActorCriticAgent(ActorCriticAgent):
    actor_critic: ExpectedActorCritic
