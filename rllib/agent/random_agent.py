"""Implementation of a random agent."""

from rllib.dataset import TrajectoryDataset
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    def __init__(
        self,
        dim_state,
        dim_action,
        num_states=-1,
        num_actions=-1,
        gamma=1,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            train_frequency=0,
            num_rollouts=0,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
        self.policy = RandomPolicy(
            dim_state, dim_action, num_states=num_states, num_actions=num_actions
        )
        self.dataset = TrajectoryDataset(sequence_length=1)

    @classmethod
    def default(
        cls,
        environment,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
        """See `AbstractAgent.default'."""
        return RandomAgent(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        self.dataset.append(self.last_trajectory)
        super().end_episode()
