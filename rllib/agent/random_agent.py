"""Implementation of a random agent."""

from rllib.dataset import TrajectoryDataset
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    def __init__(self, env_name, dim_state, dim_action, num_states=-1,
                 num_actions=-1, gamma=1, exploration_steps=0, exploration_episodes=0):
        super().__init__(env_name, train_frequency=0, num_rollouts=0, gamma=gamma,
                         exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.policy = RandomPolicy(dim_state, dim_action, num_states=num_states,
                                   num_actions=num_actions)
        self.dataset = TrajectoryDataset(sequence_length=1)

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        self.dataset.append(self.last_trajectory)
        super().end_episode()
