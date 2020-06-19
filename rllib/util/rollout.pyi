from typing import Callable, List, Optional, Tuple, Union

from numpy import ndarray
from torch import Tensor

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import (
    Action,
    Distribution,
    Observation,
    State,
    Termination,
    Trajectory,
)
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward

def step_env(
    environment: AbstractEnvironment,
    state: Union[int, ndarray],
    action: Union[int, ndarray],
    pi: Optional[Distribution] = ...,
    render: bool = ...,
    goal: Optional[Union[int, ndarray]] = ...,
) -> Tuple[Observation, Union[int, ndarray], bool, dict]: ...
def step_model(
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    termination: Termination,
    state: Tensor,
    action: Tensor,
    done: Tensor,
    pi: Optional[Distribution] = ...,
) -> Tuple[Observation, Tensor, Tensor]: ...
def record(
    environment: AbstractEnvironment,
    agent: AbstractAgent,
    save_dir: str,
    num_episodes: int = ...,
    max_steps: int = ...,
) -> None: ...
def rollout_agent(
    environment: AbstractEnvironment,
    agent: AbstractAgent,
    num_episodes: int = ...,
    max_steps: int = ...,
    render: bool = ...,
    print_frequency: int = ...,
    plot_frequency: int = ...,
    save_milestones: Optional[List[int]] = ...,
    plot_callbacks: Optional[List[Callable[[AbstractAgent, int], None]]] = ...,
) -> None: ...
def rollout_policy(
    environment: AbstractEnvironment,
    policy: AbstractPolicy,
    num_episodes: int = ...,
    max_steps: int = ...,
    render: bool = ...,
    **kwargs,
) -> List[Trajectory]: ...
def rollout_model(
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    policy: AbstractPolicy,
    initial_state: State,
    termination: Optional[Termination] = ...,
    max_steps: int = ...,
    **kwargs,
) -> Trajectory: ...
def rollout_actions(
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    action_sequence: Action,
    initial_state: State,
    termination: Optional[Termination] = ...,
) -> Trajectory: ...
