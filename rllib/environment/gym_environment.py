from .abstract_environment import AbstractEnvironment
import gym


class GymEnvironment(AbstractEnvironment):
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        try:
            dim_action = self._env.action_space.shape[0]
            num_action = None
        except IndexError:
            dim_action = 1
            num_action = self._env.action_space.n

        super().__init__(
            dim_action=dim_action,
            dim_state=self._env.observation_space.shape[0],
            action_space=self._env.action_space,
            observation_space=self._env.observation_space,
            num_action=num_action
        )
        self._time = 0

    @property
    def state(self):
        return self._env.state

    @state.setter
    def state(self, value):
        self._env.state = value

    @property
    def time(self):
        return self._time

    def step(self, action):
        self._time += 1
        return self._env.step(action)

    def reset(self):
        self._time = 0
        return self._env.reset()
