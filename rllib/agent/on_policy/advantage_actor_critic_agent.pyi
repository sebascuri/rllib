from rllib.algorithms.a2c import A2C

from .actor_critic_agent import ActorCriticAgent

class A2CAgent(ActorCriticAgent):
    algorithm: A2C
