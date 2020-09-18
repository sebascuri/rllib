"""Implementation of DQNAgent Algorithms."""
import torch.nn.modules.loss as loss

from rllib.algorithms.soft_q_learning import SoftQLearning

from .q_learning_agent import QLearningAgent


class SoftQLearningAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    temperature: float or ParameterDecay.
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

    def __init__(
        self, critic, temperature=0.2, criterion=loss.MSELoss, *args, **kwargs
    ):
        kwargs.pop("policy")
        super().__init__(
            critic=critic,
            policy=None,  # type: ignore
            criterion=criterion,
            *args,
            **kwargs,
        )
        self.algorithm = SoftQLearning(
            critic=critic,
            criterion=criterion(reduction="none"),
            temperature=temperature,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
