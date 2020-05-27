"""Implementation of TD3 Algorithm."""
from itertools import chain

from .dpg_agent import DPGAgent
from rllib.value_function import NNEnsembleQFunction


class TD3Agent(DPGAgent):
    """Abstract Implementation of the TD3 Algorithm.

    The TD3 algorithm is like the DDPG algorithm but it has an ensamble of Q-functions
    to decrease the maximization bias.

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
    Fujimoto, S., et al. (2018). Addressing Function Approximation Error in
    Actor-Critic Methods. ICML.

    """

    def __init__(self, env_name, q_function, policy, criterion,
                 optimizer, memory, exploration_noise, num_iter=1, batch_size=64,
                 target_update_frequency=4,
                 policy_noise=0., noise_clip=1.,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):

        q_function = NNEnsembleQFunction.from_q_function(q_function=q_function,
                                                         num_heads=2)
        optimizer = type(optimizer)(
            chain(policy.parameters(), q_function.parameters()), **optimizer.defaults)
        super().__init__(env_name, q_function=q_function, policy=policy,
                         exploration_noise=exploration_noise,
                         criterion=criterion,
                         optimizer=optimizer,
                         memory=memory,
                         num_iter=num_iter, batch_size=batch_size,
                         target_update_frequency=target_update_frequency,
                         policy_noise=policy_noise, noise_clip=noise_clip,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
