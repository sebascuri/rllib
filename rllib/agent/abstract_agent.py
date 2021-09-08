"""Interface for agents."""
import contextlib
from abc import ABCMeta
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from rllib.dataset.datatypes import Loss
from rllib.dataset.utilities import average_dataclass
from rllib.util.early_stopping import EarlyStopping
from rllib.util.logger import Logger
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import save_random_state, tensor_to_distribution


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

    def __init__(
        self,
        optimizer=None,
        train_frequency=0,
        num_rollouts=0,
        num_iter=1,
        batch_size=1,
        policy_update_frequency=1,
        target_update_frequency=1,
        clip_gradient_val=float("Inf"),
        early_stopping_epsilon=-1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
        training_verbose=False,
        device="cpu",
        log_dir=None,
        *args,
        **kwargs,
    ):
        self.logger = Logger(
            self.name if log_dir is None else log_dir,
            tensorboard=tensorboard,
            comment=comment,
        )
        self.early_stopping_algorithm = EarlyStopping(epsilon=early_stopping_epsilon)

        self.counters = {"total_episodes": 0, "total_steps": 0, "train_steps": 0}
        self.episode_steps = []

        self.gamma = gamma
        self.exploration_episodes = exploration_episodes
        self.exploration_steps = exploration_steps
        self.train_frequency = train_frequency
        self.num_rollouts = num_rollouts
        self.num_iter = num_iter
        self.batch_size = batch_size

        self.policy_update_frequency = policy_update_frequency
        self.target_update_frequency = target_update_frequency
        self.clip_gradient_val = clip_gradient_val
        self.optimizer = optimizer

        self.training = True
        self._training_verbose = training_verbose
        self.comment = comment
        self.last_trajectory = []
        self.params = {}
        self.device = device

    def set_policy(self, new_policy):
        """Set policy."""
        self.policy = new_policy
        self.algorithm.set_policy(new_policy)

    def to(self, device):
        """Send agent to device."""
        self.algorithm.to(device=device)

    @classmethod
    def default(cls, environment, comment=None, gamma=0.99, *args, **kwargs):
        """Get default agent for a given environment."""
        return cls(
            comment=environment.name if comment is None else comment,
            gamma=gamma,
            *args,
            **kwargs,
        )

    def __str__(self):
        """Generate string to parse the agent."""
        comment = self.comment if len(self.comment) else self.policy.__class__.__name__
        opening = "=" * 88
        str_ = (
            f"\n{opening}\n{self.name} & {comment}\n"
            f"Total episodes {self.total_episodes}\n"
            f"Total steps {self.total_steps}\n"
            f"Train steps {self.train_steps}\n"
            f"{self.logger}{opening}\n"
        )
        return str_

    def act(self, state):
        """Ask the agent for an action to interact with the environment."""
        if self.total_steps < self.exploration_steps or (
            self.total_episodes < self.exploration_episodes
        ):
            policy = self.policy.random()
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(
                    state, dtype=torch.get_default_dtype(), device=self.device
                )
            policy = self.policy(state)

        self.pi = tensor_to_distribution(policy, **self.policy.dist_params)
        if self.training:
            action = self.pi.sample()
        elif self.pi.has_enumerate_support:
            action = torch.argmax(self.pi.probs)
        else:
            try:
                action = self.pi.mean
            except NotImplementedError:
                action = self.pi.sample((100,)).mean(dim=0)

        if not self.policy.discrete_action:
            action = action.clamp(-1.0, 1.0)
            action = self.policy.action_scale * action
        return action.detach().to("cpu").numpy()

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        if self.training:
            self.policy.update()  # update policy parameters (eps-greedy.)
            self.counters["total_steps"] += 1
            self.episode_steps[-1] += 1
        self.logger.update(rewards=observation.reward.item())
        self.logger.update(entropy=observation.entropy.item())

        self.last_trajectory.append(observation)
        observation.to(self.device)

    def start_episode(self):
        """Start a new episode."""
        self.policy.reset()
        self.last_trajectory = []

        self.episode_steps.append(0)

    def set_goal(self, goal):
        """Set goal."""
        self.policy.set_goal(goal)

    def end_episode(self):
        """End an episode."""
        rewards = self.logger.current["rewards"]
        environment_return = rewards[0] * rewards[1]
        if self.training:
            self.counters["total_episodes"] += 1
            self.logger.end_episode(train_return=environment_return)
        else:
            self.logger.end_episode(eval_return=environment_return)

        # save checkpoint at every episode.
        self.save_checkpoint()

        if environment_return >= max(
            self.logger.get("train_return") + self.logger.get("eval_return")
        ):  # logger.get() returns a list!
            self.save(f"best.pkl")

    def end_interaction(self):
        """End the interaction with the environment."""
        pass

    def learn(self, *args, **kwargs):
        """Train the agent."""
        pass

    def early_stop(self, losses, **kwargs):
        """Early stop the training algorithm."""
        self.early_stopping_algorithm.update(
            losses.critic_loss.mean().item(), losses.dual_loss.mean().item()
        )
        return False

    def train(self, val=True):
        """Set the agent in training mode."""
        self.training = val

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.train(not val)

    def _learn_steps(self, closure):
        """Apply `num_iter' learn steps to closure function."""
        for _ in tqdm(range(self.num_iter), disable=not self._training_verbose):
            if self.train_steps % self.policy_update_frequency == 0:
                cm = contextlib.nullcontext()
            else:
                cm = DisableGradient(self.policy)

            with cm:
                losses = self.optimizer.step(closure=closure)  # type: Loss

            self.logger.update(**asdict(average_dataclass(losses)))
            self.logger.update(**self.algorithm.info())

            self.counters["train_steps"] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

            if self.early_stop(losses, **self.algorithm.info()):
                break
        self.algorithm.reset()
        self.early_stopping_algorithm.reset()

    @property
    def total_episodes(self):
        """Return number of steps in current episode."""
        return self.counters["total_episodes"]

    @property
    def total_steps(self):
        """Return number of steps of interaction with environment."""
        return self.counters["total_steps"]

    @property
    def train_steps(self):
        """Return number of steps of interaction with environment."""
        return self.counters["train_steps"]

    @property
    def train_at_observe(self):
        """Raise flag if train after an observation."""
        return (
            self.training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.train_frequency > 0  # train after a transition.
            and self.total_steps % self.train_frequency == 0  # correct steps.
        )

    @property
    def train_at_end_episode(self):
        """Raise flag to train at end of episode."""
        return (
            self.training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.num_rollouts > 0  # train once the episode ends.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        )

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__

    def save_checkpoint(self):
        """Save a checkpoint of the agent at the end of each episode."""
        self.logger.export_to_json()
        self.save(f"last.pkl")
        save_random_state(self.logger.log_dir)

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
            directory = self.logger.log_dir
        path = f"{directory}/{filename}"

        params = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Logger) or key == "pi":
                continue
            elif isinstance(value, nn.Module) or isinstance(value, Optimizer):
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
            if isinstance(value, Logger) or key == "pi":
                continue
            elif isinstance(value, nn.Module) or isinstance(value, Optimizer):
                value.load_state_dict(agent_dict[key])
            else:
                self.__dict__[key] = agent_dict[key]
