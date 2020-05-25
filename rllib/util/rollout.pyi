from typing import List, Union, Tuple, Callable

from numpy import ndarray

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation, State, Action, Distribution, \
    Trajectory, Termination
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward


def step(environment: AbstractEnvironment, state: Union[int, ndarray],
         action: Union[int, ndarray], pi: Distribution, render: bool,
         goal: Union[int, ndarray] = None
         ) -> Tuple[Observation, Union[int, ndarray], bool, dict]: ...


def rollout_agent(environment: AbstractEnvironment, agent: AbstractAgent,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False,
                  print_frequency: int = 0, plot_frequency: int = 0,
                  milestones: List[int] = None,
                  plot_callbacks: List[Callable[[AbstractAgent, int], None]] = None
                  ) -> None: ...


def rollout_policy(environment: AbstractEnvironment, policy: AbstractPolicy,
                   num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                   **kwargs) -> List[Trajectory]: ...


def rollout_model(dynamical_model: AbstractModel, reward_model: AbstractReward,
                  policy: AbstractPolicy, initial_state: State,
                  termination: Termination = None, max_steps: int = 1000,
                  **kwargs) -> Trajectory: ...


def rollout_actions(dynamical_model: AbstractModel, reward_model: AbstractReward,
                    action_sequence: Action, initial_state: State,
                    termination: Termination = None) -> Trajectory: ...
