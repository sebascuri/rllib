"""Implementation of SARSA Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.logger import Logger


class SARSAAgent(AbstractAgent):
    """Implementation of a SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    sarsa_algorithm: QLearning
        Implementation of Q-Learning algorithm.
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

    def __init__(self, sarsa_algorithm, q_function, policy, criterion, optimizer,
                 batch_size=1, target_update_frequency=1, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.sarsa_algorithm = sarsa_algorithm(q_function, criterion(reduction='none'),
                                               gamma)
        self.policy = policy
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self._last_observation = None
        self._batch_size = batch_size
        self._trajectory = list()

        self.logs['td_errors'] = Logger('abs_mean')
        self.logs['losses'] = Logger('mean')

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        if self._last_observation:
            self._trajectory.append(self._last_observation._replace(next_action=action))
        return action

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._last_observation = observation

        if len(self._trajectory) >= self._batch_size:
            self.train(self._trajectory)
            self._trajectory = list()
        if self.total_steps % self.target_update_frequency == 0:
            self.sarsa_algorithm.update()

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self._last_observation = None

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        # The next action is irrelevant as the next value is zero for all actions.
        action = super().act(self._last_observation.state)
        self._trajectory.append(self._last_observation._replace(next_action=action))
        self.train(self._trajectory)

        super().end_episode()

    def train(self, trajectory):
        """Train the SARSA agent using the `trajectory'.

        Parameters
        ----------
        trajectory: List[Observation]

        """
        trajectory = Observation(*stack_list_of_tuples(trajectory))

        self.optimizer.zero_grad()
        ans = self.sarsa_algorithm(
            trajectory.state, trajectory.action, trajectory.reward,
            trajectory.next_state, trajectory.done, trajectory.next_action, self.policy)

        loss = ans.loss.mean()
        loss.backward()

        self.optimizer.step()

        self.logs['td_errors'].append(ans.td_error.mean().item())
        self.logs['losses'].append(loss.item())
