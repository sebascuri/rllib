from .abstract_agent import AbstractAgent


class SarsaAgent(AbstractAgent):
    def __init__(self, q_function, exploration, gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._q_function = q_function
        self._exploration = exploration


    def act(self, state):
        logits = self._q_function(torch.tensor(state).float())
        action_distribution = Categorical(logits=logits)
        return self._exploration(action_distribution, self.total_steps)

    def observe(self, observation):
        super().observe(observation)
        self._memory.append(observation)
        if len(self._memory) >= self._hyper_params['batch_size']:
            self._train()

    def start_episode(self):
        super().start_episode()
        self.logs['td_errors'].append([])

    def end_episode(self):
        if self.num_episodes % self._hyper_params['target_update_frequency'] == 0:
            self._q_target.parameters = self._q_function.parameters

        self.logs['q_function'].append(self._q_function.state_dict())