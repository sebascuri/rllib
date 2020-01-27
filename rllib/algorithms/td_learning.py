"""Implementation of LSTD algorithm."""

from abc import ABC, abstractmethod
import torch
import copy


class ERTDLearning(ABC):
    """Implementation of TD Learning algorithms with an experience replay."""

    def __init__(self, value_function, criterion, optimizer, environment, memory,
                 gamma, target_update_frequency=1):
        self.value_function = value_function
        self.value_function_target = copy.deepcopy(value_function)
        self.criterion = criterion
        self.optimizer = optimizer
        self.environment = environment
        self.memory = memory
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency

    def train(self, batches):
        """Train using TD Learning."""
        losses = []
        td_errors = []
        for batch in range(batches):
            observation, idx, w = self.memory.get_batch()
            self.optimizer.zero_grad()

            # Experience Replay
            state, action, reward, next_state, done = observation

            # Simulator
            self.environment.state = state
            action = self.policy(torch.tensor(state).float()).sample().numpy()
            next_state, reward, done, _ = self.environment.step(action)

            pred_v, target_v = self._td(state.float(), reward.float(),
                                        next_state.float(), done.float())

            td_error = (pred_v.detach() - target_v.detach())
            td_errors.append(td_error.mean().item())

            loss = self.criterion(pred_v, target_v, reduction='none')
            loss = torch.tensor(w).float() * loss
            loss.mean().backward()

            losses.append(loss.mean().item())

            self.optimizer.step()
            self.memory.update(idx, td_error.numpy())
        return td_errors, losses

    @abstractmethod
    def _td(self, state, reward, next_state, done):
        raise NotImplementedError


class ERLSTD(ERTDLearning):
    """Experience replay LSTD algorithm."""

    def _td(self, state, reward, next_state, done):
        pred_v = self.value_function(state)
        target_v = reward + self.gamma * self.value_function(next_state) * (1 - done)

        return pred_v, target_v.detach()
