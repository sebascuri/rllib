"""Implementation of TD3 Algorithm."""
from .abstract_dpg_agent import AbstractDPGAgent
from rllib.value_function import NNEnsembleQFunction
import torch


class TD3Agent(AbstractDPGAgent):
    """Abstract Implementation of the DDPG Algorithm.

    The AbstractDDPGAgent algorithm implements the DDPT-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

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
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, q_function, policy, exploration, criterion, critic_optimizer,
                 actor_optimizer, memory, max_action=1,
                 target_update_frequency=4, policy_update_frequency=1,
                 policy_noise=0., noise_clip=1.,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):

        q_function = NNEnsembleQFunction(q_function=q_function, num_heads=2)
        params = list()
        for q in q_function.ensemble:
            params += list(q.parameters)
        critic_optimizer = type(critic_optimizer)(params, **critic_optimizer.defaults)
        super().__init__(q_function, policy, exploration, criterion, critic_optimizer,
                         actor_optimizer, memory, max_action=max_action,
                         target_update_frequency=target_update_frequency,
                         policy_update_frequency=policy_update_frequency,
                         policy_noise=policy_noise, noise_clip=noise_clip,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)

    def _train_critic(self, state, action, reward, next_state, done, weight):
        self.critic_optimizer.zero_grad()
        pred_q, target_q = self._td(state, action, reward, next_state, done)
        loss = torch.zeros_like(target_q)
        td_error = torch.zeros_like(target_q)
        for q in pred_q:
            loss += (weight * self.criterion(q, target_q, reduction='none'))
            td_error += q.detach() - target_q.detach()

        loss.mean().backward()
        self.critic_optimizer.step()

        td_error_mean = td_error.mean().item()
        self.logs['td_errors'].append(td_error_mean)
        self.logs['episode_td_errors'][-1].append(td_error_mean)
        return td_error

    def _train_actor(self, state, weight):
        self.actor_optimizer.zero_grad()
        action = self.policy(state).mean.clamp(-1, 1)
        q = -self.q_function[0](state.float(), action)
        loss = weight * q
        loss.mean().backward()
        self.actor_optimizer.step()

    def _td(self, state, action, reward, next_state, done):
        pred_q = self.q_function(state, action)

        next_policy_action = self.policy_target(next_state).mean
        next_action_noise = (torch.randn_like(next_policy_action) * self.policy_noise
                             ).clamp(-self.noise_clip, self.noise_clip)
        next_policy_action = (next_policy_action + next_action_noise).clamp(-1, 1)

        next_v = torch.min(*self.q_target(next_state, next_policy_action))
        target_q = reward + self.gamma * next_v * (1 - done)

        return pred_q, target_q.detach()
