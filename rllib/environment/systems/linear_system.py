"""Implementation of a Linear Dynamical System."""


import numpy as np
from .abstract_system import AbstractSystem


__all__ = ['LinearSystem']


class LinearSystem(AbstractSystem):
    """An environment Discrete Time for Linear Dynamical Systems.

    Parameters
    ----------
    a: ndarray
        state transition matrix.
    b: ndarray
        input matrix.
    c: ndarray, optional
        observation matrix.
    """

    def __init__(self, a, b, c=None):
        self.a = np.atleast_2d(a)
        self.b = np.atleast_2d(b)
        if c is None:
            c = np.eye(self.a.shape[0])
        self.c = c

        dim_state, dim_action = self.b.shape
        dim_observation = self.c.shape[0]

        super().__init__(dim_state=dim_state,
                         dim_action=dim_action,
                         dim_observation=dim_observation,
                         )
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def reset(self, state=None):
        self._time = 0
        self.state = np.atleast_1d(state)
        return self.c @ self.state

    def step(self, action):
        action = np.atleast_1d(action)
        self.state = self.a @ self.state + self.b @ action
        return self.c @ self.state
