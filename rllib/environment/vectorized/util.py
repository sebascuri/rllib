"""Utilities for vectorized environments.."""
from abc import ABCMeta

import numpy as np
from gym import Env

from rllib.util.utilities import get_backend


class VectorizedEnv(Env, metaclass=ABCMeta):
    """Vectorized implementation of Acrobot."""

    @property
    def bk(self):
        """Get current backend of environment."""
        return get_backend(self.state)

    def atan2(self, sin, cos):
        """Return signed angle of the sin cosine."""
        if self.bk is np:
            return self.bk.arctan2(sin, cos)
        else:
            return self.bk.atan2(sin, cos)

    def clip(self, val, min_val, max_val):
        """Clip between min and max values."""
        if self.bk is np:
            return self.bk.clip(val, min_val, max_val)
        else:
            return self.bk.clamp(val, min_val, max_val)

    def cat(self, arrays, axis=-1):
        """Concatenate arrays along an axis."""
        if self.bk is np:
            return np.append(*arrays, axis)
        else:
            return self.bk.cat(arrays, axis)

    def rand(self, min_val, max_val):
        """Return random number between min_val and max_val."""
        if self.bk is np:
            return np.random.randn() * (max_val - min_val) + min_val
        else:
            return self.bk.rand(min_val, max_val)

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()` to
        reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action: np.ndarray
            An action provided by the agent.

        Returns
        -------
        observation: np.ndarray
            Agent's observation of the current environment.
        reward: float
            Amount of reward returned after previous action.
        done: bool
            Whether the episode has ended.
        info: dict
            Contains auxiliary diagnostic information.
        """
        raise NotImplementedError


def rk4(derivs, y0, t, *args, **kwargs):
    """Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.

    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """
    bk = get_backend(y0)
    yout = bk.zeros((len(t), *y0.shape))

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = derivs(y0, thist, *args, **kwargs)
        k2 = derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs)
        k3 = derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs)
        k4 = derivs(y0 + dt * k3, thist + dt, *args, **kwargs)
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
