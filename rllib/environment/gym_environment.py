"""Wrapper for OpenAI-Gym Environments."""

import gym
import gym.envs.atari
import gym.wrappers

from .abstract_environment import AbstractEnvironment
from .utilities import parse_space


class GymEnvironment(AbstractEnvironment):
    """Wrapper for OpenAI-Gym Environments.

    Parameters
    ----------
    env_name: str
        environment name
    seed: int, optional
        random seed to initialize environment.

    """

    def __init__(self, env_name, seed=None, **kwargs):
        env = gym.make(env_name, **kwargs)
        if isinstance(env, gym.wrappers.TimeLimit) and not kwargs.get(
            "episodic", False
        ):
            env = env.unwrapped
        if isinstance(env, gym.envs.atari.AtariEnv):
            env = gym.wrappers.AtariPreprocessing(env)
        self.env = env
        self.env.seed(seed)

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0
        self.metadata = self.env.metadata

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        self._time += 1
        next_state, reward, done, info = self.env.step(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        return next_state, reward, done, info

    def render(self, mode="human"):
        """See `AbstractEnvironment.render'."""
        return self.env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self.env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self.env.reset()

    @property
    def goal(self):
        """Return current goal of environment."""
        if hasattr(self.env, "goal"):
            return self.env.goal
        return None

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        if hasattr(self.env, "state"):
            return self.env.state
        elif hasattr(self.env, "s"):
            return self.env.s
        elif hasattr(self.env, "_get_obs"):
            return getattr(self.env, "_get_obs")()
        else:
            raise NotImplementedError("Strange state")

    @state.setter
    def state(self, value):
        if hasattr(self.env, "state"):
            self.env.state = value
        elif hasattr(self.env, "s"):
            self.env.s = value
        elif hasattr(self.env, "set_state"):
            self.env.set_state(
                value[: len(self.env.sim.data.qpos)],
                value[len(self.env.sim.data.qpos) :],
            )
        else:
            raise NotImplementedError("Strange state")

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.env.__class__.__name__
