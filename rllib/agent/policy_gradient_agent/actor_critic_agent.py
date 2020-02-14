"""Implementation of REINFORCE Algorithm."""

from .abstract_policy_gradient_agent import AbstractPolicyGradient


class ActorCritic(AbstractPolicyGradient):
    """Implementation of ACTOR-CRITIC algorithm.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def _value_estimate(self, trajectories):
        values = []
        for trajectory in trajectories:
            values.append(self.critic(trajectory.state, trajectory.action))

        return values

    def _td_base(self, state, action, reward, next_state, done, value_estimate=None):
        next_v = self.baseline(next_state) * (1 - done)
        target_v = reward + self.gamma * next_v
        return self.baseline(state), target_v.detach()

    def _td_critic(self, state, action, reward, next_state, done):
        pred_q = self.critic(state, action)

        next_policy_action = self.policy_target(next_state).sample()
        next_v = self.critic_target(next_state, next_policy_action) * (1 - done)
        target_q = reward + self.gamma * next_v

        return pred_q, target_q.detach()


class ExpectedActorCritic(ActorCritic):
    """Implementation of ACTOR-CRITIC algorithm.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def _td_critic(self, state, action, reward, next_state, done):
        pred_q = self.critic(state, action)
        next_v = self.critic_target.value(next_state, self.policy) * (1 - done)
        target_q = reward + self.gamma * next_v
        return pred_q, target_q.detach()


class AdvantageActorCritic(AbstractPolicyGradient):

    def _value_estimate(self, trajectories):
        values = []
        for trajectory in trajectories:
            r = trajectory.reward
            next_v = self.critic(trajectory.next_state) * (1 - trajectory.done)
            v = self.critic(trajectory.state)
            values.append(r + self.gamma * next_v - v)
        return values

    def _td_base(self, state, action, reward, next_state, done, value_estimate=None):
        next_v = self.baseline(next_state) * (1 - done)
        target_v = reward + self.gamma * next_v
        return self.baseline(state), target_v.detach()

    def _td_critic(self, state, action, reward, next_state, done):
        pred_q = self.critic(state)
        next_v = self.critic_target(next_state) * (1 - done)
        target_q = reward + self.gamma * next_v
        return pred_q, target_q.detach()
