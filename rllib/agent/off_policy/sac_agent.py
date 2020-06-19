"""Implementation of DQNAgent Algorithms."""
from rllib.algorithms.sac import SoftActorCritic
from rllib.value_function import NNEnsembleQFunction

from .off_policy_agent import OffPolicyAgent


class SACAgent(OffPolicyAgent):
    """Implementation of a SAC agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    memory: ExperienceReplay
        Memory where to store the observations.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a
    stochastic actor. ICML.

    """

    def __init__(
        self,
        q_function,
        policy,
        criterion,
        optimizer,
        memory,
        eta,
        regularization=False,
        num_iter=1,
        batch_size=64,
        target_update_frequency=4,
        train_frequency=1,
        num_rollouts=0,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        q_function = NNEnsembleQFunction.from_q_function(
            q_function=q_function, num_heads=2
        )
        self.algorithm = SoftActorCritic(
            policy=policy,
            q_function=q_function,
            criterion=criterion(reduction="none"),
            gamma=gamma,
            eta=eta,
            regularization=regularization,
        )

        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **optimizer.defaults,
        )
        super().__init__(
            optimizer=optimizer,
            memory=memory,
            batch_size=batch_size,
            num_iter=num_iter,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
        self.policy = self.algorithm.policy
        self.dist_params = self.algorithm.dist_params
