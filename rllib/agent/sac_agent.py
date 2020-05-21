"""Implementation of DQNAgent Algorithms."""
import torch

from rllib.agent.off_policy_ac_agent import OffPolicyACAgent
from rllib.algorithms.sac import SoftActorCritic
from rllib.value_function import NNEnsembleQFunction
from rllib.util import tensor_to_distribution


class SACAgent(OffPolicyACAgent):
    """Implementation of a SAC agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    memory: ExperienceReplay
        Memory where to store the observations.
    temperature: ParameterDecay.
        Temperature of Soft Q function.
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
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a
    stochastic actor. ICML.

    """

    def __init__(self, env_name, q_function, policy,
                 criterion, critic_optimizer, actor_optimizer, memory, temperature,
                 num_iter=1, batch_size=64,
                 target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1., train_frequency=1, num_rollouts=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):

        q_function = NNEnsembleQFunction.from_q_function(q_function=q_function,
                                                         num_heads=2)
        critic_optimizer = type(critic_optimizer)(q_function.parameters(),
                                                  **critic_optimizer.defaults)
        super().__init__(env_name, actor_optimizer, critic_optimizer, memory,
                         batch_size=batch_size, num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         policy_update_frequency=policy_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.algorithm = SoftActorCritic(policy, q_function,
                                         criterion(reduction='none'), temperature,
                                         gamma)
        self.policy = self.algorithm.policy

    def act(self, state):
        """See `AbstractAgent.act'."""
        if self.total_steps < self.exploration_steps or (
                self.total_episodes < self.exploration_episodes):
            policy = self.policy.random()
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.get_default_dtype())
            policy = self.policy(state)

        self.pi = tensor_to_distribution(policy, tanh=True,
                                         action_scale=self.policy.action_scale)
        if self._training:
            action = self.pi.sample()
        else:
            self.pi = tensor_to_distribution(policy, tanh=False)
            action = self.pi.mean

        action = action.detach().numpy().clip(-self.policy.action_scale.numpy(),
                                              self.policy.action_scale.numpy())
        return action
