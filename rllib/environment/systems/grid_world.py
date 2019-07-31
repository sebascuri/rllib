from .abstract_system import AbstractSystem
from gym import spaces


class GridWorld(AbstractSystem):
    def __init__(self, grid_size, obstacles=None):
        num_dim = len(grid_size)
        super().__init__(num_dim, num_dim, num_dim)
        self._num_actions = [2] * num_dim
        self._num_observations = grid_size
        self._grid_size = grid_size
        self._state = self.observation_space.sample()
        if obstacles is None:
            obstacles = []
        self._obstacles = obstacles
        self._time = 0

    def reset(self, state):
        self._state = state
        self._time = 0

    def step(self, action):
        next_state = self._state + (-1 + 2 * action)
        if next_state in self._obstacles:
            next_state = self._state

        larger_idx = next_state >= self._grid_size
        smaller_idx = next_state < 0
        for idx in [larger_idx, smaller_idx]:
            next_state[idx] = self._state[idx]

        self._state = next_state
        return self._state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def time(self):
        return self._time

    @property
    def action_space(self):
        return spaces.MultiDiscrete(self._num_actions)

    @property
    def observation_space(self):
        return spaces.MultiDiscrete(self._num_observations)
