"""Project Data-types."""
from collections import namedtuple
from typing import Callable, List, NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import Delta
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal, Uniform

Array = Union[np.ndarray, torch.Tensor]
State = Union[int, float, Array]
Action = Union[int, float, Array]
Reward = Union[int, float, Array]
Probability = Union[int, float, Array]
Done = Union[bool, Array]
Gaussian = Union[MultivariateNormal, Delta]
Distribution = Union[MultivariateNormal, Delta, Categorical, Uniform]
TupleDistribution = Union[Tensor, Tuple[Tensor, Tensor]]

Termination = Union[nn.Module, Callable[[State, Action, State], Done]]
NaN = float("nan")

Observation = namedtuple(
    "Observation",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "next_action",
        "log_prob_action",
        "entropy",
        "state_scale_tril",
        "next_state_scale_tril",
    ],
)


class RawObservation(NamedTuple):
    """Observation datatype."""

    state: State
    action: Action
    reward: Reward = torch.tensor(NaN)
    next_state: State = torch.tensor(NaN)
    done: Done = False
    next_action: Action = torch.tensor(NaN)  # SARSA algorithm.
    log_prob_action: Probability = torch.tensor(NaN)  # Off-policy algorithms.
    entropy: Probability = torch.tensor(NaN)  # Entropy of current policy.
    state_scale_tril: Tensor = torch.tensor(NaN)
    next_state_scale_tril: Tensor = torch.tensor(NaN)

    @staticmethod
    def _is_equal_nan(x, y):
        x, y = np.array(x), np.array(y)
        if ((np.isnan(x)) ^ (np.isnan(y))).all():  # XOR
            return False
        elif ((~np.isnan(x)) & (~np.isnan(y))).all():
            return np.allclose(x, y)
        else:
            return True

    @staticmethod
    def get_example(
        dim_state: int = 1,
        dim_action: int = 1,
        num_states: int = -1,
        num_actions: int = -1,
        kind: str = "zero",
    ):
        """Get example observation.

        Parameters
        ----------
        dim_state: int, optional (default=1).
            State dimension.
        dim_action: int, optional (default=1).
            Action dimension.
        num_states: int, optional (default=-1).
            Number of states, if discrete.
        num_actions: int, optional (default=-1).
            Number of actions, if discrete.
        kind: str, optional (default='zero').
            Kind of example. Options are ['zero', 'random', 'nan']
        """
        if kind not in ["zero", "random", "nan"]:
            raise ValueError(f"{kind} not in ['zero', 'random', 'nan'].")

        discrete_state = num_states >= 0
        if discrete_state:
            state = torch.randint(num_states, (1,))
            next_state = torch.randint(num_states, (1,))
        else:
            state = torch.randn(dim_state)
            next_state = torch.randn(dim_state)

        discrete_action = num_actions >= 0
        if discrete_action:
            action = torch.randint(num_actions, (1,))
            next_action = torch.randint(num_actions, (1,))
            log_prob_action = torch.tensor(1.0)

        else:
            action = torch.randn(dim_action)
            next_action = torch.randn(dim_action)
            log_prob_action = torch.ones(dim_action)

        if kind == "random":
            pass
        elif kind == "zero":
            state = 0 * state
            next_state = 0 * next_state
            action = 0 * action
            next_action = 0 * next_action

        elif kind == "nan":
            state = NaN * state
            next_state = NaN * next_state
        else:
            raise NotImplementedError

        return Observation(
            state=state,
            action=action,
            reward=torch.rand(1)[0],
            next_state=next_state,
            done=torch.tensor(False),
            next_action=next_action,
            log_prob_action=log_prob_action,
            entropy=torch.tensor(1.0),
            state_scale_tril=torch.tensor(NaN),
            next_state_scale_tril=torch.tensor(NaN),
        )

    @staticmethod
    def nan_example(
        dim_state: int = 1,
        dim_action: int = 1,
        num_states: int = -1,
        num_actions: int = -1,
    ):
        """Return a NaN Example."""
        return RawObservation.get_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            kind="nan",
        )

    @staticmethod
    def zero_example(
        dim_state: int = 1,
        dim_action: int = 1,
        num_states: int = -1,
        num_actions: int = -1,
    ):
        """Return a Zero Example."""
        return RawObservation.get_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            kind="zero",
        )

    @staticmethod
    def random_example(
        dim_state: int = 1,
        dim_action: int = 1,
        num_states: int = -1,
        num_actions: int = -1,
    ):
        """Return a Random Example."""
        return RawObservation.get_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            kind="random",
        )

    def __eq__(self, other):
        """Check if two observations are equal."""
        if not isinstance(other, Observation):
            return NotImplemented
        is_equal = self._is_equal_nan(self.state, other.state)
        is_equal &= self._is_equal_nan(self.action, other.action)
        is_equal &= self._is_equal_nan(self.reward, other.reward)
        is_equal &= self._is_equal_nan(self.next_state, other.next_state)
        is_equal &= self._is_equal_nan(self.next_action, other.next_action)
        is_equal &= self._is_equal_nan(self.log_prob_action, other.log_prob_action)
        is_equal &= self._is_equal_nan(self.done, other.done)
        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not self.__eq__(other)

    def to_torch(self):
        """Transform to torch."""
        return Observation(
            *map(
                lambda x: x
                if isinstance(x, torch.Tensor)
                else torch.tensor(x, dtype=torch.get_default_dtype()),
                self,
            )
        )


Trajectory = List[Observation]
