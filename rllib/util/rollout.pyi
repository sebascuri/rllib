from rllib.agent.abstract_agent import AbstractAgent
from rllib.policy.abstract_policy import AbstractPolicy
from rllib.environment.abstract_environment import AbstractEnvironment
from rllib.model.abstract_model import AbstractModel
from typing import List, Union, Tuple, Callable
from rllib.dataset.datatypes import Observation, State, Action, StateAction
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

def rollout_model(model: AbstractModel, policy: AbstractPolicy, initial_states: State,
                  max_steps: int = 1000,
                  termination: Callable[[State, Action], bool] = None) -> StateAction: ...
