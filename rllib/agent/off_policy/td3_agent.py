"""Implementation of TD3 Algorithm."""
from rllib.util.parameter_decay import Constant
from rllib.value_function import NNEnsembleQFunction

from .dpg_agent import DPGAgent


class TD3Agent(DPGAgent):
    """Abstract Implementation of the TD3 Algorithm.

    The TD3 algorithm is like the DDPG algorithm but it has an ensamble of Q-functions
    to decrease the maximization bias.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: AbstractPolicy
        policy that is learned.
    exploration_noise: ParameterDecay.
        exploration strategy that returns the actions.

    References
    ----------
    Fujimoto, S., et al. (2018). Addressing Function Approximation Error in
    Actor-Critic Methods. ICML.

    """

    def __init__(self, critic, policy, *args, **kwargs):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)
        self.optimizer = type(self.optimizer)(
            [
                p
                for n, p in self.algorithm.named_parameters()
                if "target" not in n and "old_policy" not in n
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, critic=None, exploration_noise=None, *args, **kwargs):
        """Get Default TD3 agent."""
        if critic is None:
            critic = NNEnsembleQFunction.default(environment)
        if exploration_noise is None:
            noise = Constant(0.1)
        return super().default(
            environment, critic=critic, exploration_noise=noise, *args, **kwargs
        )
