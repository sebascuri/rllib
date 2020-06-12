"""Interface for agents."""

from abc import ABCMeta

import torch
import torch.nn as nn
import torch.optim as optim

from rllib.util.logger import Logger
from rllib.util.utilities import tensor_to_distribution


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

    def __init__(self, train_frequency, num_rollouts,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0,
                 tensorboard=False, comment=''):
        self.logger = Logger(
            self.name,
            tensorboard=tensorboard,
            comment=comment
        )
        self.counters = {'total_episodes': 0, 'total_steps': 0, 'train_steps': 0}
        self.episode_steps = []

        self.gamma = gamma
        self.exploration_episodes = exploration_episodes
        self.exploration_steps = exploration_steps
        self.train_frequency = train_frequency
        self.num_rollouts = num_rollouts

        self._training = True

        self.comment = comment
        self.last_trajectory = []
        self.params = {}
        self.dist_params = {}

    def __str__(self):
        """Generate string to parse the agent."""
        comment = self.comment if len(self.comment) else self.policy.__class__.__name__
        opening = "=" * 88
        str_ = f"\n{opening}\n{self.name} with {comment}\n"
        str_ += f"Total episodes {self.total_episodes}\n"
        str_ += f"Total steps {self.total_steps}\n"
        str_ += f"Train steps {self.train_steps}\n"
        str_ += f"{self.logger}{opening}\n"
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

        self.pi = tensor_to_distribution(policy, **self.dist_params)
        if self._training:
            action = self.pi.sample()
        elif self.pi.has_enumerate_support:
            action = torch.argmax(self.pi.probs)
        else:
            try:
                action = self.pi.mean
            except NotImplementedError:
                action = self.pi.sample((100,)).mean(dim=0)

        return action.detach().numpy()

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        self.policy.update()  # update policy parameters (eps-greedy.)

        self.counters['total_steps'] += 1
        self.episode_steps[-1] += 1
        self.logger.update(rewards=observation.reward.item())
        self.logger.update(entropy=observation.entropy.item())

        self.last_trajectory.append(observation)

    def start_episode(self, **kwargs):
        """Start a new episode."""
        self.policy.reset(**kwargs)
        self.episode_steps.append(0)
        self.last_trajectory = []

    def end_episode(self):
        """End an episode."""
        self.counters['total_episodes'] += 1
        rewards = self.logger.current['rewards']
        self.logger.end_episode(environment_return=rewards[0] * rewards[1])
        self.logger.export_to_json()  # save at every episode?

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
    def train_steps(self):
        """Return number of steps of interaction with environment."""
        return self.counters['train_steps']

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__

    def save(self, filename, directory=None):
        """Save agent.

        Parameters
        ----------
        filename: str.
            Filename with which to save the agent.
        directory: str, optional.
            Directory where to save the agent. By default use the log directory.

        Returns
        -------
        path: str.
            Path where agent is saved.
        """
        if directory is None:
            directory = self.logger.writer.logdir
        path = f"{directory}/{filename}"

        params = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Logger):
                pass
            elif isinstance(value, nn.Module) or isinstance(value, optim.Optimizer):
                params[key] = value.state_dict()
            else:
                params[key] = value

        torch.save(params, path)
        return path

    def load(self, path):
        """Load agent.

        Parameters
        ----------
        path: str.
            Full path to agent.
        """
        agent_dict = torch.load(path)

        for key, value in self.__dict__.items():
            if isinstance(value, Logger):
                pass
            elif isinstance(value, nn.Module) or isinstance(value, optim.Optimizer):
                value.load_state_dict(agent_dict[key])
            else:
                self.__dict__[key] = agent_dict[key]
