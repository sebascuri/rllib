"""Implementation of Episodic Q Actor-Critic Algorithm."""

from abc import ABCMeta
from .abstract_actor_critic_agent import AbstractPolicyGradient
from rllib.value_function import AbstractQFunction
from rllib.util.utilities import integrate


class AbstractExpectedActorCritic(AbstractPolicyGradient, metaclass=ABCMeta):
    """Abstract Implementation of Expected Actor-Critic algorithm.

    This Actor critic algorithm updates at the end of each episode only.

    References
    ----------
    Ciosek, K., & Whiteson, S. (2018). Expected policy gradients. AAAI.
    """

    critic: AbstractQFunction
    critic_target: AbstractQFunction

    def _train_actor(self, trajectories):
        self.policy_optimizer.zero_grad()
        for trajectory in trajectories:
            pi = self.policy(trajectory.state)
            loss = -integrate(
                lambda a: pi.log_prob(a) * self._return(
                    trajectory.state, action=a, reward=trajectory.reward,
                    next_state=trajectory.next_state, done=trajectory.done), pi)
            loss.sum().backward()

        self.policy_optimizer.step()
