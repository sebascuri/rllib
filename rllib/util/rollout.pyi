from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.reward import AbstractReward
from typing import List, Union, Tuple, Callable
from rllib.dataset.datatypes import Observation, State, Action
from numpy import ndarray


def _step(environment: AbstractEnvironment, state: Union[int, ndarray],
          action: Union[int, ndarray], render: bool
          ) -> Tuple[Observation, Union[int, ndarray], bool]:...


def rollout_agent(environment: AbstractEnvironment, agent: AbstractAgent,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False,
                  milestones: List[int] = None
                  ) -> None: ...

def rollout_policy(environment: AbstractEnvironment, policy: AbstractPolicy,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                  ) -> List[List[Observation]]: ...

def rollout_model(dynamical_model: AbstractModel, reward_model: AbstractReward, policy: AbstractPolicy,
                  initial_state: State, termination: Callable[[State, Action], bool] = None,
                  max_steps: int = 1000) -> List[Observation]: ...
