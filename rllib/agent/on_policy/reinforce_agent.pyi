from rllib.algorithms.reinforce import REINFORCE

from .on_policy_agent import OnPolicyAgent

class REINFORCEAgent(OnPolicyAgent):
    algorithm: REINFORCE
