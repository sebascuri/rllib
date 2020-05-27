"""Implementation of Deterministic Policy Gradient Algorithms."""

from .off_policy_agent import OffPolicyAgent
from rllib.algorithms.dpg import DPG
from rllib.util.parameter_decay import ParameterDecay, Constant


class DPGAgent(OffPolicyAgent):
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
    optimizer: nn.Optimizer
        q_function optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, env_name, q_function, policy, criterion,
                 optimizer, memory, exploration_noise, num_iter=1,
                 batch_size=64, target_update_frequency=4,
                 policy_noise=0., noise_clip=1.,
                 train_frequency=1, num_rollouts=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         optimizer=optimizer,
                         memory=memory, batch_size=batch_size,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        assert policy.deterministic, "Policy must be deterministic."
        self.algorithm = DPG(q_function, policy, criterion(reduction='none'), gamma,
                             policy_noise, noise_clip
                             )
        self.policy = self.algorithm.policy

        if not isinstance(exploration_noise, ParameterDecay):
            exploration_noise = Constant(exploration_noise)

        self.params['exploration_noise'] = exploration_noise
        self.dist_params = {'add_noise': True,
                            'policy_noise': self.params['exploration_noise']}

    def train(self, val=True):
        """Set the agent in training mode."""
        super().train(val)
        self.dist_params = {'add_noise': True,
                            'policy_noise': self.params['exploration_noise']}

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        super().eval(val)
        self.dist_params = {'add_noise': False,
                            'policy_noise': self.params['exploration_noise']}
