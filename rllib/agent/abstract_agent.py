from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    An agent interacts with the environment through the 'act' method.

    The agent is responsible for exploring, storing past observations, and training.

    The agent could be:
        model-free or model-based,
        on-policy or off-policy,
        random or deterministic,
        policy based or value based.

    The agent public methods are:
        act
        observe
        reset
        start_episode
        end_episode

    """
    @abstractmethod
    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        Parameters
        ----------
        state: ndarray

        Returns
        -------
        action: ndarray

        """
        raise NotImplementedError

    @abstractmethod
    def observe(self, state, action, reward, next_state):
        """Observe transition from the environment.

        Parameters
        ----------
        state: ndarray
        action: ndarray
        reward: float
        next_state: ndarray

        Returns
        -------
        None

        """
        raise NotImplementedError

    @abstractmethod
    def start_episode(self):
        """Start a new episode.

        Returns
        -------
        None

        """
        raise NotImplementedError

    @abstractmethod
    def end_episode(self):
        """End an episode.

        Returns
        -------
        None

        """
        raise NotImplementedError

    @abstractmethod
    def end_interaction(self):
        """End the interaction with the environment.

        Returns
        -------
        None
        """
        raise NotImplementedError
