"""Implementation of TD3 Algorithm."""
from itertools import chain

from rllib.value_function import NNEnsembleQFunction

from .dpg_agent import DPGAgent


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
    exploration_noise: ParameterDecay.
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

    def __init__(
        self,
        q_function,
        policy,
        criterion,
        optimizer,
        memory,
        exploration_noise,
        *args,
        **kwargs,
    ):
        q_function = NNEnsembleQFunction.from_q_function(
            q_function=q_function, num_heads=2
        )
        optimizer = type(optimizer)(
            chain(policy.parameters(), q_function.parameters()), **optimizer.defaults
        )
        super().__init__(
            q_function=q_function,
            policy=policy,
            exploration_noise=exploration_noise,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            *args,
            **kwargs,
        )
