from rllib.agent.abstract_agent import AbstractAgent
from rllib.policy.abstract_policy import AbstractPolicy
from rllib.environment.abstract_environment import AbstractEnvironment
from typing import List
from rllib.dataset import Observation

def rollout_agent(environment: AbstractEnvironment, agent: AbstractAgent,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False,
                  milestones: list = None
                  ) -> None: ...

def rollout_policy(environment: AbstractEnvironment, policy: AbstractPolicy,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                  ) -> List[List[Observation]]: ...