"""Interface for agents."""


from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """Interface for agents that interact with an environment.

    Parameters
    ----------
    gamma: float, optional (default=1.0)
        MDP discount factor.
    episode_length: int, optional (default=None)
        maximum number of interaction steps in a single episode.

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

    def __init__(self, gamma=1.0, episode_length=None):
        self.num_episodes = 0
        self.logs = {'total_steps': 0,
                     'episode_steps': [],
                     'rewards': [],
                     'episode_rewards': []}
        self.gamma = gamma
        self.episode_length = episode_length if episode_length else float('+Inf')

        self.training = True

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
        self.logs['total_steps'] += 1
        self.logs['episode_steps'][-1] += 1
        self.logs['episode_rewards'][-1] += observation.reward
        self.logs['rewards'][-1].append(observation.reward)

    def start_episode(self):
        """Start a new episode."""
        self.num_episodes += 1

        self.logs['episode_steps'].append(0)
        self.logs['episode_rewards'].append(0)
        self.logs['rewards'].append([])

    def end_episode(self):
        """End an episode."""
        pass

    def end_interaction(self):
        """End the interaction with the environment."""
        pass

    @property
    def episodes_steps(self):
        """Return number of steps in current episode."""
        return self.logs['episode_steps']

    @property
    def episodes_rewards(self):
        """Return rewards in all the episodes seen."""
        return self.logs['rewards']

    @property
    def episodes_cumulative_rewards(self):
        """Return cumulative rewards in each episodes."""
        return self.logs['episode_rewards']

    @property
    def total_steps(self):
        """Return number of steps of interaction with environment."""
        return self.logs['total_steps']

    @property
    def episode_steps(self):
        """Return total steps of interaction with environment in current episode."""
        return self.logs['episode_steps'][-1]

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__

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

    def train(self, mode=True):
        """Set the module in training mode."""
        self.training = mode

    def eval(self):
        """Set the module in training mode."""
        self.train(mode=False)
