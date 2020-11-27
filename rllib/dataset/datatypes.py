"""Project Data-types."""
from dataclasses import dataclass, field
from typing import List, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

Array = Union[np.ndarray, torch.Tensor]
State = Union[int, float, Array]
Action = Union[int, float, Array]
Reward = Union[int, float, Array]
Probability = Union[int, float, Array]
Done = Union[bool, Array]
TupleDistribution = Union[Tensor, Tuple[Tensor, Tensor]]

NaN = float("nan")

T = TypeVar("T", bound="Observation")


@dataclass
class Observation:
    """Observation datatype."""

    state: State
    action: Action = torch.tensor(NaN)
    reward: Reward = torch.tensor(NaN)
    next_state: State = torch.tensor(NaN)
    done: Done = False
    next_action: Action = torch.tensor(NaN)  # SARSA algorithm.
    log_prob_action: Probability = torch.tensor(NaN)  # Off-policy algorithms.
    entropy: Probability = torch.tensor(NaN)  # Entropy of current policy.
    state_scale_tril: Tensor = torch.tensor(NaN)
    next_state_scale_tril: Tensor = torch.tensor(NaN)
    reward_scale_tril: Tensor = torch.tensor(NaN)

    def __iter__(self):
        """Iterate the properties of the observation."""
        yield from self.__dict__.values()

    @staticmethod
    def _is_equal_nan(x, y):
        x, y = np.array(x), np.array(y)
        if ((np.isnan(x)) ^ (np.isnan(y))).all():  # XOR
            return False
        elif ((~np.isnan(x)) & (~np.isnan(y))).all():
            return np.allclose(x, y)
        else:
            return True

    @classmethod
    def get_example(
        cls: Type[T],
        dim_state: Tuple = (1,),
        dim_action: Tuple = (1,),
        num_states: int = -1,
        num_actions: int = -1,
        kind: str = "zero",
    ) -> T:
        """Get example observation.

        Parameters
        ----------
        dim_state: Tuple, optional (default=(1,)).
            State dimension.
        dim_action: Tuple, optional (default=(1,)).
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
            state = 1.0 * torch.randint(num_states, (1,))[0]
            next_state = 1.0 * torch.randint(num_states, (1,))[0]
        else:
            state = torch.randn(dim_state)
            next_state = torch.randn(dim_state)

        discrete_action = num_actions >= 0
        if discrete_action:
            action = 1.0 * torch.randint(num_actions, ())
            log_prob_action = torch.tensor(1.0)

        else:
            action = torch.randn(dim_action)
            log_prob_action = torch.tensor(1.0)

        reward = torch.rand(1)[0]
        done = torch.round(torch.rand(()))
        if kind == "zero":
            state = 0 * state
            next_state = 0 * next_state
            action = 0 * action
            reward = 0 * reward
            done = torch.tensor(1.0)
        elif kind == "nan":
            state = NaN * state
            next_state = NaN * next_state
            action = NaN * action
            reward = NaN * reward
            done = torch.tensor(1.0)
        elif kind != "random":
            raise NotImplementedError(f"{kind} not implemented.")

        return cls(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_action=torch.tensor(NaN),
            log_prob_action=log_prob_action,
            entropy=torch.tensor(0.0),
            state_scale_tril=torch.tensor(NaN),
            next_state_scale_tril=torch.tensor(NaN),
            reward_scale_tril=torch.tensor(NaN),
        )

    @classmethod
    def nan_example(
        cls: Type[T],
        dim_state: Tuple = (1,),
        dim_action: Tuple = (1,),
        num_states: int = -1,
        num_actions: int = -1,
    ) -> T:
        """Return a NaN Example."""
        return cls.get_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            kind="nan",
        )

    @classmethod
    def zero_example(
        cls: Type[T],
        dim_state: Tuple = (1,),
        dim_action: Tuple = (1,),
        num_states: int = -1,
        num_actions: int = -1,
    ) -> T:
        """Return a Zero Example."""
        return cls.get_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            kind="zero",
        )

    @classmethod
    def random_example(
        cls: Type[T],
        dim_state: Tuple = (1,),
        dim_action: Tuple = (1,),
        num_states: int = -1,
        num_actions: int = -1,
    ) -> T:
        """Return a Random Example."""
        return cls.get_example(
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

    def clone(self):
        """Get a cloned copy of the current observation."""
        return Observation(*tuple(x.clone() for x in self.to_torch()))

    def to(self, *args, **kwargs):
        """Perform dtypes and device conversions. See torch.to()."""
        return Observation(*tuple(x.to(*args, **kwargs) for x in self.to_torch()))

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


@dataclass
class Loss:
    """Basic Loss class.

    Other Parameters
    ----------------
    losses: Tensor.
        Combined loss to optimize.
    td_error: Tensor.
        TD-Error of critic.
    policy_loss: Tensor.
        Loss of policy optimization.
    reg_loss: Tensor.
        Either KL-divergence or entropy bonus.
    dual_loss: Tensor.
        Loss of dual minimization problem.
    """

    combined_loss: torch.Tensor = field(init=False)
    td_error: torch.Tensor = torch.tensor(0.0)
    policy_loss: torch.Tensor = torch.tensor(0.0)
    critic_loss: torch.Tensor = torch.tensor(0.0)
    reg_loss: torch.Tensor = torch.tensor(0.0)
    dual_loss: torch.Tensor = torch.tensor(0.0)

    def __post_init__(self):
        """Post-initialize Loss dataclass.

        Fill in the attribute `loss' by adding all other losses.
        """
        self.combined_loss = (
            self.policy_loss + self.critic_loss + self.reg_loss + self.dual_loss
        )

    def __add__(self, other):
        """Add two losses."""
        return Loss(*map(lambda x: x[0] + x[1], zip(self, other)))

    def __sub__(self, other):
        """Add two losses."""
        return Loss(*map(lambda x: x[0] - x[1], zip(self, other)))

    def __neg__(self):
        """Substract two losses."""
        return Loss(*map(lambda x: -x, self))

    def __mul__(self, other):
        """Multiply losses by a scalar."""
        return Loss(*map(lambda x: x * other, self))

    def __rmul__(self, other):
        """Multiply losses by a scalar."""
        return self * other

    def __truediv__(self, other):
        """Divide losses by a scalar."""
        return Loss(*map(lambda x: x / other, self))

    def __iter__(self):
        """Iterate through the losses and yield all the separated losses.

        Notes
        -----
        It does not return the loss entry.
        It is useful to create new losses.
        """
        for key, value in self.__dict__.items():
            if key == "combined_loss":
                continue
            else:
                yield value

    def reduce(self, kind):
        """Reduce losses."""
        if kind == "sum":
            return Loss(*map(lambda x: x.sum(), self))
        elif kind == "mean":
            return Loss(*map(lambda x: x.mean(), self))
        elif kind == "none":
            return self
        else:
            raise NotImplementedError


Trajectory = List[Observation]
