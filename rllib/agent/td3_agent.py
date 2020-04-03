"""Implementation of TD3 Algorithm."""
from rllib.agent import DPGAgent
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
    critic_optimizer: nn.Optimizer
        q_function optimizer.
    actor_optimizer: nn.Optimizer
        policy optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Fujimoto, S., et al. (2018). Addressing Function Approximation Error in
    Actor-Critic Methods. ICML.

    """

    def __init__(self, environment, q_function, policy, exploration, criterion,
                 critic_optimizer, actor_optimizer, memory, num_iter=1, batch_size=64,
                 max_action=1, target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1.,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):

        q_function = NNEnsembleQFunction.from_q_function(q_function=q_function,
                                                         num_heads=2)
        critic_optimizer = type(critic_optimizer)(q_function.parameters(),
                                                  **critic_optimizer.defaults)
        super().__init__(environment, q_function, policy, exploration, criterion,
                         critic_optimizer, actor_optimizer, memory,
                         num_iter=num_iter, batch_size=batch_size,
                         max_action=max_action,
                         target_update_frequency=target_update_frequency,
                         policy_update_frequency=policy_update_frequency,
                         policy_noise=policy_noise, noise_clip=noise_clip,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
