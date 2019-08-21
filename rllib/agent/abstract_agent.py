"""Interface for agents."""


from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """Interface for agents that interact with an environment.

    Methods
    -------
    act(state): int or ndarray
        Given a state, it returns an action to input to the environment.
    observe(observation):
        Record an observation from the environment.
    start_episode:
        Start a new episode.
    end_episode:
        End an episode.
    end_interaction:
        End an interaction with an environment.
    policy:
        Return the policy the agent is using.

    """

    def __init__(self, episode_length=None):
        self._steps = {'total': 0, 'episode': 0}
        self._num_episodes = 0
        self._statistics = {'episode_steps': [],
                            'rewards': [],
                            'episode_rewards': []}
        self.episode_length = episode_length if episode_length else float('+Inf')

    @abstractmethod
    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        Parameters
        ----------
        state: int or ndarray

        Returns
        -------
        action: int or ndarray

        """
        raise NotImplementedError

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        self._steps['total'] += 1
        self._steps['episode'] += 1

        self._statistics['episode_steps'][-1] += 1
        self._statistics['episode_rewards'][-1] += observation.reward
        self._statistics['rewards'][-1].append(observation.reward)

    def start_episode(self):
        """Start a new episode."""
        self._steps['episode'] = 0
        self._num_episodes += 1

        self._statistics['episode_steps'].append(0)
        self._statistics['episode_rewards'].append(0)
        self._statistics['rewards'].append([])

    def end_episode(self):
        """End an episode."""
        pass

    def end_interaction(self):
        """End the interaction with the environment."""
        pass

    @property
    def episodes_steps(self):
        """Return number of steps in current episode."""
        return self._statistics['episode_steps']

    @property
    def episodes_rewards(self):
        """Return rewards in all the episodes seen."""
        return self._statistics['rewards']

    @property
    def episodes_cumulative_rewards(self):
        """Return cumulative rewards in each episodes."""
        return self._statistics['episode_rewards']

    @property
    def total_steps(self):
        """Return number of steps of interaction with environment."""
        return self._steps['total']

    @property
    def episode_steps(self):
        """Return total steps of interaction with environment in current episode."""
        return self._steps['episode']

    @property
    def num_episodes(self):
        """Return number of episodes the agent interacted with the environment."""
        return self._num_episodes

    @property
    @abstractmethod
    def policy(self):
        """Return the policy the agent is currently using.

        Returns
        -------
        policy: AbstractPolicy

        Raises
        ------
        NotImplementedError:
            If a policy cannot be extracted.

        """
        raise NotImplementedError
