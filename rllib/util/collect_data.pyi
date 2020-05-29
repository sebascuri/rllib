"""Utilities to collect data."""
from typing import List, Union

from rllib.dataset.datatypes import Distribution, Observation
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward


def collect_environment_transitions(state_dist: Distribution,
                                    policy: Union[Distribution, AbstractPolicy],
                                    environment: AbstractEnvironment,
                                    num_samples: int) -> List[Observation]: ...


def collect_model_transitions(state_dist: Distribution,
                              policy: Union[Distribution, AbstractPolicy],
                              dynamical_model: AbstractModel,
                              reward_model: AbstractReward,
                              num_samples: int) -> List[Observation]: ...
