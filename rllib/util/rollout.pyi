from rllib.agent.abstract_agent import AbstractAgent
from rllib.policy.abstract_policy import AbstractPolicy
from rllib.environment.abstract_environment import AbstractEnvironment
from typing import List, Union, Tuple
from rllib.dataset import Observation
from numpy import ndarray


def _step(environment: AbstractEnvironment, state: Union[int, ndarray],
          action: Union[int, ndarray], render: bool
          ) -> Tuple[Observation, Union[int, ndarray], bool]:...


def rollout_agent(environment: AbstractEnvironment, agent: AbstractAgent,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False,
                  milestones: list = None
                  ) -> None: ...

def rollout_policy(environment: AbstractEnvironment, policy: AbstractPolicy,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                  ) -> List[List[Observation]]: ...