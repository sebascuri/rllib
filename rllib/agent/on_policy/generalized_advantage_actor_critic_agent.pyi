from rllib.algorithms.gaac import GAAC

from .actor_critic_agent import ActorCriticAgent

class GAACAgent(ActorCriticAgent):
    algorithm: GAAC
