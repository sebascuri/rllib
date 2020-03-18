"""Interface for agents."""

from abc import ABCMeta
import torch
import numpy as np
from rllib.util.logger import Logger


class AbstractAgent(object, metaclass=ABCMeta):
    """Interface for agents that interact with an environment.

    Parameters
    ----------
    gamma: float, optional (default=1.0)
        MDP discount factor.
    exploration_steps: int, optional (default=0)
        initial exploratory steps.
    exploration_episodes: int, optional (default=0)
        initial exploratory episodes

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
    """

    def __init__(self, gamma=1.0, exploration_steps=0, exploration_episodes=0):
        self.logs = {'rewards': Logger('sum'), 'policy entropy': Logger('mean')}
        self.counters = {'total_episodes': 0, 'total_steps': 0}
        self.episode_steps = []

        self.gamma = gamma
        self.exploration_episodes = exploration_episodes
        self.exploration_steps = exploration_steps

        self._training = True

    def __repr__(self):
        """Generate string to parse the agent."""
        opening = "====================================\n"
        str_ = opening
        str_ += "Total episodes {}\n".format(self.counters['total_episodes'])
        str_ += "Total steps {}\n".format(self.counters['total_steps'])
        rewards = self.logs['rewards'].episode_log
        entropy = self.logs['policy entropy'].episode_log
        str_ += "Average reward {:.1f}\n".format(np.mean(rewards))
        str_ += "10-Episode reward {:.1f}\n".format(np.mean(rewards[-10:]))
        if not self.policy.deterministic:
            str_ += "Policy entropy {:.2e}\n".format(np.mean(entropy))
            str_ += "10-Episode policy entropy {:.2e}\n".format(np.mean(entropy[-10:]))
        str_ += opening
        return str_

    def act(self, state):
        """Ask the agent for an action to interact with the environment."""
        if self.total_steps < self.exploration_steps or (
                self.total_episodes < self.exploration_episodes):
            policy = self.policy.random()
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.get_default_dtype())
            policy = self.policy(state)

        if self._training:
            action = policy.sample()
        else:
            if policy.has_enumerate_support:
                action = torch.argmax(policy.probs)
            else:
                action = policy.mean

        if not self.policy.deterministic:
            self.logs['policy entropy'].append(policy.entropy().detach().numpy())
        return action.detach().numpy()

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        observation = observation.to_torch()
        self.policy.update(observation)  # update policy parameters (eps-greedy.)

        self.counters['total_steps'] += 1
        self.episode_steps[-1] += 1
        self.logs['rewards'].append(observation.reward)

    def start_episode(self):
        """Start a new episode."""
        self.counters['total_episodes'] += 1
        self.episode_steps.append(0)
        for log in self.logs.values():
            log.start_episode()

    def end_episode(self):
        """End an episode."""
        for log in self.logs.values():
            log.end_episode()

    def end_interaction(self):
        """End the interaction with the environment."""
        pass

    def _train(self):
        """Train the agent."""
        pass

    def train(self, val=True):
        """Set the agent in training mode."""
        self._training = val

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.train(not val)

    @property
    def total_episodes(self):
        """Return number of steps in current episode."""
        return self.counters['total_episodes']

    @property
    def total_steps(self):
        """Return number of steps of interaction with environment."""
        return self.counters['total_steps']

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__
