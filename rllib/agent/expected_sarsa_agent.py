"""Implementation of Expected SARSA Agent."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.esarsa import ESARSA
from rllib.dataset.utilities import stack_list_of_tuples


class ExpectedSARSAAgent(AbstractAgent):
    """Implementation of an Expected SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    batch_size: int
        Number of trajectory batches before performing a TD-pdate.
    optimizer: nn.optim
        Optimization algorithm for q_function.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(self, environment, q_function, policy, criterion, optimizer,
                 num_iter=1, batch_size=1, target_update_frequency=1, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(environment, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.sarsa = ESARSA(q_function, criterion(reduction='none'), policy, gamma)
        self.policy = policy
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.trajectory = []

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectory.append(observation)
        if len(self.trajectory) >= self.batch_size:
            if self._training:
                self._train()
            self.trajectory = []

        if self.total_steps % self.target_update_frequency == 0:
            self.sarsa.update()

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if len(self.trajectory) and self._training:
            self._train()
        super().end_episode()

    def _train(self):
        """Train the SARSA agent."""
        trajectory = stack_list_of_tuples(self.trajectory)

        for _ in range(self.num_iter):
            # Update critic
            self.optimizer.zero_grad()
            losses = self.sarsa(trajectory.state, trajectory.action, trajectory.reward,
                                trajectory.next_state, trajectory.done)

            loss = losses.loss.mean()
            loss.backward()
            self.optimizer.step()

            # Update loss
            self.logger.update(critic_losses=loss.item(),
                               td_errors=losses.td_error.abs().mean().item())
