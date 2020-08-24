"""Implementation of DQNAgent Algorithms."""
import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.soft_q_learning import SoftQLearning
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.value_function import NNQFunction

from .q_learning_agent import QLearningAgent


class SoftQLearningAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    temperature: ParameterDecay.
        Temperature of Soft Q function.

    References
    ----------
    Fox, R., Pakman, A., & Tishby, N. (2015).
    Taming the noise in reinforcement learning via soft updates. UAI.

    Schulman, J., Chen, X., & Abbeel, P. (2017).
    Equivalence between policy gradients and soft q-learning.

    Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017).
    Reinforcement learning with deep energy-based policies. ICML.

    O'Donoghue, B., Munos, R., Kavukcuoglu, K., & Mnih, V. (2016).
    Combining policy gradient and Q-learning. ICLR.
    """

    def __init__(self, q_function, criterion, temperature, *args, **kwargs):

        super().__init__(
            q_function=q_function,
            policy=None,  # type: ignore
            criterion=criterion,
            *args,
            **kwargs,
        )
        self.algorithm = SoftQLearning(
            critic=q_function,
            criterion=criterion(reduction="none"),
            temperature=temperature,
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = NNQFunction(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            tau=5e-3,
            input_transform=None,
        )
        optimizer = Adam(q_function.parameters(), lr=3e-4)
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        return cls(
            q_function=q_function,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            temperature=0.2,
            num_iter=1,
            batch_size=100,
            target_update_frequency=1,
            train_frequency=1,
            num_rollouts=0,
            comment=environment.name,
            *args,
            **kwargs,
        )
