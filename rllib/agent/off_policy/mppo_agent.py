"""MPPO Agent Implementation."""
from .off_policy_agent import OffPolicyAgent
from rllib.algorithms.mppo import MPPO


class MPPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPPO."""

    def __init__(self, env_name,
                 policy, q_function,
                 optimizer, memory,
                 criterion,
                 num_action_samples=15,
                 entropy_reg=0.,
                 epsilon=None, epsilon_mean=None, epsilon_var=None,
                 eta=None, eta_mean=None, eta_var=None,
                 num_iter=100, batch_size=64,
                 target_update_frequency=4,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):

        self.algorithm = MPPO(policy=policy, q_function=q_function,
                              num_action_samples=num_action_samples,
                              criterion=criterion, entropy_reg=entropy_reg,
                              epsilon=epsilon, epsilon_mean=epsilon_mean,
                              epsilon_var=epsilon_var,
                              eta=eta, eta_mean=eta_mean, eta_var=eta_var, gamma=gamma)

        self.policy = self.algorithm.policy
        optimizer = type(optimizer)([p for name, p in self.algorithm.named_parameters()
                                     if 'target' not in name], **optimizer.defaults)
        super().__init__(env_name, memory=memory, optimizer=optimizer,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         batch_size=batch_size,
                         train_frequency=train_frequency,
                         num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes,
                         comment=comment)
