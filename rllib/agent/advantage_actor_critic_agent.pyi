from rllib.algorithms.a2c import A2C
from .actor_critic_agent import ActorCriticAgent


class A2CAgent(ActorCriticAgent):
    actor_critic: A2C
