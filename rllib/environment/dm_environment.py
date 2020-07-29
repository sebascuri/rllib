"""Wrapper for DMSuite Environments."""

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from .abstract_environment import AbstractEnvironment

try:
    import glfw
    from dm_control import suite

    glfw.init()

    class DMSuiteEnvironment(AbstractEnvironment):
        """Wrapper for DM-Suite Environments.

        Parameters
        ----------
        env_name: str.
            environment name
        env_task: str.
            environment task.
        seed: int, optional.
            random seed to initialize environment with.

        """

        def __init__(self, env_name, env_task, seed=None, **kwargs):
            self.env = suite.load(env_name, task_name=env_task, task_kwargs=kwargs)
            self.env.task.random.seed(seed)
            self._name = f"{env_name} {env_task}"

            action_spec = self.env.action_spec()
            dim_action = action_spec.shape
            action_space = Box(low=action_spec.minimum, high=action_spec.maximum)

            observation_spec = self.env.observation_spec()
            dim_state = 0
            for s in observation_spec.values():
                try:
                    dim_state += s.shape[0]
                except IndexError:
                    pass
            observation_space = Box(
                low=-np.inf * np.ones(dim_state), high=np.inf * np.ones(dim_state)
            )

            super().__init__(
                dim_action=dim_action,
                dim_state=(dim_state,),
                action_space=action_space,
                observation_space=observation_space,
                num_actions=-1,
                num_states=-1,
                num_observations=-1,
            )
            self._time = 0

        @staticmethod
        def _stack_observations(observations):
            out = []
            for value in observations.values():
                out.append(value)
            return np.concatenate(out)

        def step(self, action):
            """See `AbstractEnvironment.step'."""
            self._time += 1
            dm_obs = self.env.step(action)
            reward = dm_obs.reward
            return self._stack_observations(dm_obs.observation), reward, False, {}

        def render(self, mode="human"):
            """See `AbstractEnvironment.render'."""
            plt.imshow(self.env.physics.render())
            plt.show(block=False)
            plt.pause(0.01)

        def close(self):
            """See `AbstractEnvironment.close'."""
            self.env.close()

        def reset(self):
            """See `AbstractEnvironment.reset'."""
            self._time = 0
            dm_obs = self.env.reset()
            return self._stack_observations(dm_obs.observation)

        @property
        def goal(self):
            """Return current goal of environment."""
            if hasattr(self.env, "goal"):
                return self.env.goal
            return None

        @property
        def state(self):
            """See `AbstractEnvironment.state'."""
            return self.env.task.get_observation(self.env.physics)

        @state.setter
        def state(self, value):
            self.env.task.set_state(value)

        @property
        def time(self):
            """See `AbstractEnvironment.time'."""
            return self._time

        @property
        def name(self):
            """Return environment name."""
            return self._name


except Exception:  # dm_control not installed.
    pass
