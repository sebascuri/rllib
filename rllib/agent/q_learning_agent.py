from .abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader

__all__ = ['QLearningAgent', 'GradientQLearningAgent', 'DeepQLearningAgent',
           'DoubleDQNAgent']


class AbstractQLearningAgent(AbstractAgent):
    def __init__(self, q_function, q_target, exploration, criterion, optimizer, memory,
                 hyper_params):
        self._q_function = q_function
        self._q_target = q_target
        self._exploration = exploration
        self._criterion = criterion
        self._memory = memory
        self._hyper_params = hyper_params
        self._optimizer = optimizer(self._q_function.parameters(),
                                    lr=self._hyper_params['learning_rate'])

        self._data_loader = DataLoader(self._memory,
                                       batch_size=self._hyper_params['batch_size'])
        self._steps = {'total': 0, 'episode': 0, 'num_episodes': 0}
        self._statistics = {'episode_steps': [],
                            'rewards': [],
                            'episode_rewards': []}

    def act(self, state):
        self._steps['total'] += 1
        self._steps['episode'] += 1
        logits = self._q_function(torch.from_numpy(state).float())
        action_distribution = Categorical(logits)
        return self._exploration(action_distribution, self._steps['total'])

    def observe(self, observation):
        self._memory.append(observation)
        i_episode = self._steps['num_episodes']
        self._statistics['episode_steps'][i_episode] += 1
        self._statistics['episode_rewards'][i_episode] += observation.reward
        self._statistics['rewards'][i_episode].append(observation.reward)

        if len(self._memory) >= self._hyper_params['batch_size']:
            self._train()

    def start_episode(self):
        self._steps['episode'] = 0

        self._statistics['episode_steps'].append(0)
        self._statistics['episode_rewards'].append(0)
        self._statistics['rewards'].append([])

    def end_episode(self):
        self._steps['num_episodes'] += 1
        if self._steps['num_episodes'] % self._hyper_params['target_update'] == 0:
            self._q_target.load_state_dict(self._q_function.state_dict())

    def end_interaction(self):
        print(self._statistics['episode_steps'])

    def _train(self, steps=1):
        self._memory.shuffle()
        for i, observation in enumerate(self._data_loader):
            state, action, reward, next_state, done = observation
            pred_q, target_q = self._td(state.float(), action, reward.float(),
                                        next_state.float(), done.float())

            loss = self._criterion(pred_q, target_q)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if i + 1 == steps:
                break

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError


class QLearningAgent(AbstractQLearningAgent):
    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state)
        pred_q = pred_q.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # target = max Q(x', a) and stop gradient.
        target_q = self._q_function(next_state).max(dim=-1)[0]
        target_q = reward + self._hyper_params['gamma'] * target_q * (1 - done)

        return pred_q, target_q


class GradientQLearningAgent(AbstractQLearningAgent):
    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state)
        pred_q = pred_q.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # target = max Q(x', a) and stop gradient.
        target_q = self._q_function(next_state).max(dim=-1)[0]
        target_q = reward + self._hyper_params['gamma'] * target_q * (1 - done)

        return pred_q, target_q.detach()


class DeepQLearningAgent(AbstractQLearningAgent):
    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state)
        pred_q = pred_q.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # target = max Q_target(x', a)
        target_q = self._q_target(next_state).max(dim=-1)[0]
        target_q = reward + self._hyper_params['gamma'] * target_q * (1 - done)

        return pred_q, target_q.detach()


class DoubleDQNAgent(AbstractQLearningAgent):
    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state)
        pred_q = pred_q.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # target = Q_target(x', argmax Q(x', a))

        next_action = self._q_function(next_state).argmax(dim=-1)
        target_q = self._q_target(next_state)
        target_q = target_q.gather(1, next_action.unsqueeze(-1)).squeeze(-1)
        target_q = reward + self._hyper_params['gamma'] * target_q * (1 - done)

        return pred_q, target_q.detach()
