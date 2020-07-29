"""Implementation of a Class that constructs an dynamic system from an ode."""

import numpy as np
import torch
from scipy import integrate, signal

from .abstract_system import AbstractSystem
from .linear_system import LinearSystem


class ODESystem(AbstractSystem):
    """A class that constructs an dynamic system from an ode.

    Parameters
    ----------
    func : callable.
        func is the right hand side of as xdot = f(t, x).
        with actions, extend x to be states, actions.
    step_size : float
    dim_state: Tuple
    dim_action : Tuple
    """

    def __init__(
        self, func, step_size, dim_state, dim_action, integrator=integrate.RK45
    ):
        super().__init__(dim_state=dim_state, dim_action=dim_action)

        self.step_size = step_size
        self.func = func
        self._state = np.zeros(dim_state)
        self._time = 0
        self.integrator = integrator

    def step(self, action):
        """See `AbstractSystem.step'."""
        integrator = self.integrator(
            lambda t, y: self.func(t, y, action), 0, self.state, t_bound=self.step_size
        )

        while integrator.status == "running":
            integrator.step()
        self.state = integrator.y
        self._time += self.step_size

        return self.state

    def reset(self, state=None):
        """See `AbstractSystem.reset'."""
        self.state = state
        return self.state

    def linearize(self, state=None, action=None):
        """Linearize at a current state and action using torch autograd.

        By default, it considers the linearization at state=0, action=0.

        Parameters
        ----------
        state: State, optional.
            State w.r.t. which linearize. By default linearize at zero.
        action: Action, optional.
            Action w.r.t. which linearize. By default linearize at zero.

        Returns
        -------
        sys: LinearSystem
        """
        if state is None:
            state = np.zeros(self.dim_state)
        if action is None:
            action = np.zeros(self.dim_action)

        if type(state) is not torch.Tensor:
            state = torch.tensor(state)
        if type(action) is not torch.Tensor:
            action = torch.tensor(action)

        state.requires_grad = True
        action.requires_grad = True
        f = self.func(None, state, action)

        a = np.zeros((self.dim_state, self.dim_state))
        b = np.zeros((self.dim_state, self.dim_action))
        for i in range(self.dim_state):
            aux = torch.autograd.grad(
                f[i], state, allow_unused=True, retain_graph=True
            )[0]
            if aux is not None:
                a[i] = aux.numpy()

            aux = torch.autograd.grad(
                f[i], action, allow_unused=True, retain_graph=True
            )[0]
            if aux is not None:
                b[i] = aux.numpy()

        ad, bd, _, _, _ = signal.cont2discrete(
            (a, b, 0, 0), self.step_size, method="zoh"
        )
        return LinearSystem(ad, bd)

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def time(self):
        """See `AbstractSystem.time'."""
        return self._time
