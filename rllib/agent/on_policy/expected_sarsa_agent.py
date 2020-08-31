"""Implementation of Expected SARSA Agent."""

from rllib.algorithms.esarsa import ESARSA

from .sarsa_agent import SARSAAgent


class ExpectedSARSAAgent(SARSAAgent):
    """Implementation of an Expected SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(self, critic, policy, *args, **kwargs):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)
        self.algorithm = ESARSA(
            critic=critic,
            criterion=self.algorithm.criterion,
            policy=policy,
            gamma=self.gamma,
        )
        self.policy = policy
