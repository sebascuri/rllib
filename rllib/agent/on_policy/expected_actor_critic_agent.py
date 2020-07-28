"""Implementation of Expected-Actor Critic Agent."""

from rllib.algorithms.eac import ExpectedActorCritic

from .actor_critic_agent import ActorCriticAgent


class ExpectedActorCriticAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Ciosek, K., & Whiteson, S. (2018).
    Expected policy gradients. AAAI.
    """

    def __init__(
        self,
        policy,
        critic,
        optimizer,
        criterion,
        num_iter=1,
        target_update_frequency=1,
        train_frequency=0,
        num_rollouts=1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            criterion=criterion,
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
        self.algorithm = ExpectedActorCritic(
            policy, critic, criterion(reduction="none"), gamma
        )
        self.policy = self.algorithm.policy
