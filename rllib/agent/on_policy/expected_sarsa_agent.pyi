from rllib.algorithms.esarsa import ESARSA

from .on_policy_agent import OnPolicyAgent

class ExpectedSARSAAgent(OnPolicyAgent):
    algorithm: ESARSA
