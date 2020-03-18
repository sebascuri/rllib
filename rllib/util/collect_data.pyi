"""Utilities to collect data."""
from rllib.dataset.datatypes import Observation, Distribution
from rllib.policy import AbstractPolicy
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.reward import AbstractReward
from typing import List, Union


def collect_environment_transitions(state_dist: Distribution,
                                    policy: Union[Distribution, AbstractPolicy],
                                    environment: AbstractEnvironment,
                                    num_samples: int) -> List[Observation]: ...


def collect_model_transitions(state_dist: Distribution,
                              policy: Union[Distribution, AbstractPolicy],
                              dynamic_model: AbstractModel, reward_model:AbstractReward,
                              num_samples: int) -> List[Observation]: ...
