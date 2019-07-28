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

    @property
    def state(self):
        return self.ode.y

    @state.setter
    def state(self, value):
        self.ode.set_initial_value(value, t=self.ode.t)

    @property
    def time(self):
        return self.ode.t

    def reset(self, state=None):
        self.ode.set_initial_value(state, t=0.)
        return self.state

    def step(self, action):
        self.ode.set_f_params(action)
        self.ode.integrate(self.ode.t + self._step_size)
        return self.state
