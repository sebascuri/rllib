"""Implementation of a Class that constructs an dynamic system from an ode."""

from .abstract_system import AbstractSystem
from scipy import integrate


class ODESystem(AbstractSystem):
    """A class that constructs an dynamic system from an ode.

    Parameters
    ----------
    ode : callable.
        See scipy.integrate.ode for details
    step_size : float
    dim_state: int
    dim_action : int
    integrator : string
        See scipy.integrate.ode for details.
    jac: callable
        See scipy.integrate.ode for details.
    """

    def __init__(self, ode, step_size, dim_state, dim_action,
                 integrator='dopri5', jac=None,
                 ):
        super().__init__(dim_state=dim_state,
                         dim_action=dim_action,
                         )

        self._step_size = step_size
        self.ode = integrate.ode(ode, jac=jac)
        self.ode.set_integrator(integrator)

    def step(self, action):
        """See `AbstractSystem.step'."""
        self.ode.set_f_params(action)
        self.ode.integrate(self.ode.t + self._step_size)
        return self.state

    def reset(self, state=None):
        """See `AbstractSystem.reset'."""
        self.ode.set_initial_value(state, t=0.)
        return self.state

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        return self.ode.y

    @state.setter
    def state(self, value):
        self.ode.set_initial_value(value, t=self.ode.t)

    @property
    def time(self):
        """See `AbstractSystem.time'."""
        return self.ode.t
