"""MPPO Agent Implementation."""

from rllib.agent.off_policy_agent import OffPolicyAgent


class MPPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPPO."""

    def __init__(self, env_name, mppo, optimizer, memory, num_iter=100, batch_size=64,
                 train_frequency=100,
                 target_update_frequency=4,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(env_name, memory=memory,
                         batch_size=batch_size,
                         train_frequency=train_frequency,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)

        self.algorithm = mppo
        self.policy = self.algorithm.policy
        self.optimizer = optimizer
        self.target_update_frequency = target_update_frequency
        self.num_iter = num_iter

    def _train(self):
        for _ in range(self.num_iter):
            obs, idx, weight = self.memory.get_batch(self.batch_size)
            self.optimizer.zero_grad()
            losses = self.algorithm(obs.state, obs.action, obs.reward, obs.next_state,
                                    obs.done)
            losses.loss.backward()
            self.optimizer.step()
            self.logger.update(**losses._asdict())

            self.train_iter += 1
            if self.train_iter % self.target_update_frequency == 0:
                self.algorithm.update()

        self.algorithm.reset()
