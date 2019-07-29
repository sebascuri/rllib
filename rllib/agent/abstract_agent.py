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
    def __init__(self):
        self._steps = {'total': 0, 'episode': 0}
        self._num_episodes = 0
        self._statistics = {'episode_steps': [],
                            'rewards': [],
                            'episode_rewards': []}

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

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

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        Returns
        -------
        None

        """
        self._steps['total'] += 1
        self._steps['episode'] += 1

        self._statistics['episode_steps'][-1] += 1
        self._statistics['episode_rewards'][-1] += observation.reward
        self._statistics['rewards'][-1].append(observation.reward)

    def start_episode(self):
        """Start a new episode.

        Returns
        -------
        None

        """
        self._steps['episode'] = 0
        self._num_episodes += 1

        self._statistics['episode_steps'].append(0)
        self._statistics['episode_rewards'].append(0)
        self._statistics['rewards'].append([])

    def end_episode(self):
        """End an episode.

        Returns
        -------
        None

        """
        pass

    def end_interaction(self):
        """End the interaction with the environment.

        Returns
        -------
        None
        """
        pass

    @property
    def episodes_steps(self):
        return self._statistics['episode_steps']

    @property
    def episodes_rewards(self):
        return self._statistics['rewards']

    @property
    def episodes_cumulative_rewards(self):
        return self._statistics['episode_rewards']

    @property
    def total_steps(self):
        return self._steps['total']

    @property
    def episode_steps(self):
        return self._steps['episode']

    @property
    def num_episodes(self):
        return self._num_episodes
