"""Implementation of Expected SARSA Agent."""

from rllib.algorithms.esarsa import ESARSA

from .on_policy_agent import OnPolicyAgent


class ExpectedSARSAAgent(OnPolicyAgent):
    """Implementation of an Expected SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    batch_size: int
        Number of trajectory batches before performing a TD-pdate.
    optimizer: nn.optim
        Optimization algorithm for q_function.
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
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(
        self,
        q_function,
        policy,
        criterion,
        optimizer,
        num_iter=1,
        batch_size=1,
        target_update_frequency=1,
        train_frequency=1,
        num_rollouts=0,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            optimizer=optimizer,
            num_iter=num_iter,
            batch_size=batch_size,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            target_update_frequency=target_update_frequency,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
        self.algorithm = ESARSA(q_function, criterion(reduction="mean"), policy, gamma)
        self.policy = policy
