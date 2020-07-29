"""Interface for Environments."""

from abc import ABCMeta, abstractmethod

from gym.spaces import Box


class AbstractEnvironment(object, metaclass=ABCMeta):
    """Interface for Environments.

    Parameters
    ----------
    dim_state: Tuple
        dimension of state.
    dim_action: Tuple
        dimension of action.
    observation_space: gym.env.Spaces
    action_space: gym.env.Spaces
    dim_observation: int, optional
        dimension of observation.
    num_observations: int, optional
        number of discrete observations (None if observation is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Methods
    -------
    step(action): next_state, reward, done, info
        execute a step in the environment.
    reset(): reset the environment.

    """

    def __init__(
        self,
        dim_state,
        dim_action,
        observation_space,
        action_space,
        dim_observation=-1,
        num_states=-1,
        num_actions=-1,
        num_observations=-1,
    ):
        super().__init__()
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.num_actions = num_actions if num_actions is not None else -1
        self.num_observations = num_observations if num_observations is not None else -1
        self.num_states = num_states if num_states is not None else -1

        if dim_observation == -1:
            dim_observation = dim_state
        self.dim_observation = dim_observation

        self.action_space = action_space
        self.observation_space = observation_space

        self.discrete_state = self.num_states >= 0
        self.discrete_action = self.num_actions >= 0
        self.discrete_observation = self.num_observations >= 0

        self.metadata = {"render.modes": []}

    def __str__(self):
        """Return string that explains environment."""
        if self.discrete_state:
            state_str = f"{self.num_states} discrete states"
        else:
            state_str = f"{self.dim_state} continuous states"
        if self.discrete_action:
            action_str = f"{self.num_actions} discrete actions"
        else:
            action_str = f"{self.dim_action} continuous actions"

        return f"{self.name}, {state_str}, {action_str}."

    @abstractmethod
    def step(self, action):
        """Run one time-step of the model dynamics.

        Parameters
        ----------
        action: ndarray

        Returns
        -------
        observation: ndarray
        reward: float
        done: bool
        info: dict

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the state of the model and returns an initial observation.

        Returns
        -------
        observation: ndarray

        """
        raise NotImplementedError

    def render(self, mode="human"):
        """Render the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Parameters
        ----------
            mode: str.
                The mode to render with

        Note
        ----
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        """
        pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    @property
    def action_scale(self):
        """Return the action scale of the environment."""
        if self.discrete_action:
            return 1
        elif isinstance(self.action_space, Box):
            return 1 / 2 * (self.action_space.high - self.action_space.low)
        else:
            raise NotImplementedError

    @property
    def goal(self):
        """Return current goal of environment."""
        return None

    @property  # type: ignore
    @abstractmethod
    def state(self):
        """Return current state of environment."""
        raise NotImplementedError

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self):
        """Return current time of environment."""
        raise NotImplementedError

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__
