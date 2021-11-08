"""Python Script Template."""
from gym import Wrapper

from .locomotion import LocomotionEnv


class RemoveWrapper(Wrapper):
    """Remove dim_pos coordinates from the observation vector."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env, LocomotionEnv)
        self.dim_pos = self.env.dim_pos

    def reset(self, **kwargs):
        """Remove dim_pos coordinates from the observation vector and reset."""
        observation = self.env.reset()
        return observation[self.dim_pos :]

    def step(self, action):
        """Remove dim_pos coordinates from the observation vector and execute step."""
        observation, reward, done, info = self.env.step(action)
        return observation[self.dim_pos :], reward, done, info
