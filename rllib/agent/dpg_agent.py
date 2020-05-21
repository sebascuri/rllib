"""Implementation of Deterministic Policy Gradient Algorithms."""

from rllib.agent.off_policy_ac_agent import OffPolicyACAgent
from rllib.algorithms.dpg import DPG


class DPGAgent(OffPolicyACAgent):
    """Implementation of the Deterministic Policy Gradient Agent.

    The AbstractDDPGAgent algorithm implements the DPG-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

    TODO: build compatible q-function approximation.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: AbstractPolicy
        policy that is learned.
    exploration: AbstractExplorationStrategy.
        exploration strategy that returns the actions.
    criterion: nn.Module
    critic_optimizer: nn.Optimizer
        q_function optimizer.
    actor_optimizer: nn.Optimizer
        policy optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, env_name, q_function, policy, exploration, criterion,
                 critic_optimizer, actor_optimizer, memory, num_iter=1,
                 batch_size=64, target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1.,
                 train_frequency=1, num_rollouts=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         actor_optimizer=actor_optimizer,
                         critic_optimizer=critic_optimizer,
                         memory=memory, batch_size=batch_size,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         policy_update_frequency=policy_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        assert policy.deterministic, "Policy must be deterministic."
        self.algorithm = DPG(q_function, policy, criterion(reduction='none'), gamma,
                             policy_noise, noise_clip
                             )
        self.policy = self.algorithm.policy
        self.exploration = exploration

    def act(self, state):
        """See `AbstractAgent.act'.

        As the policy is deterministic, some noise must be added to aid exploration.
        """
        action = super().act(state)
        if self._training:
            action += self.exploration().numpy()

        return action.clip(-self.policy.action_scale.numpy(),
                           self.policy.action_scale.numpy())
