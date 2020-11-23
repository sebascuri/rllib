"""Model Ensemble Early Stopping Algorithm."""
import math

import numpy as np
import torch

from rllib.model.utilities import PredictionStrategy
from rllib.util.early_stopping import EarlyStopping
from rllib.util.multiprocessing import run_parallel_returns

from .abstract_mb_algorithm import AbstractMBAlgorithm


class ModelEnsembleEarlyStopping(AbstractMBAlgorithm, EarlyStopping):
    """Model Ensemble Early Stopping Algorithm.

    Evaluate the policy by simulating all the models separately.
    Return stop when a the returns in a fraction of the models increases.

    TODO: Why sample different models?
    TODO: Add discounts :) and maybe terminal returns?

    References
    ----------
    Kurutach, T., Clavera, I., Duan, Y., Tamar, A., & Abbeel, P. (2018).
    Model-ensemble trust-region policy optimization. ICLR.
    """

    def __init__(
        self,
        policy,
        fraction=0.3,
        eval_frequency=5,
        epsilon=-1.0,
        non_decrease_iter=np.inf,
        relative=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        try:
            num_models = self.dynamical_model.base_model.num_heads
        except AttributeError:
            num_models = 1
        self.num_models = num_models
        self.fraction = math.ceil(fraction * num_models)
        self.eval_frequency = eval_frequency
        self.step_counts = 0
        self.policy = policy
        self.model_ensemble_early_stopping = [
            EarlyStopping(epsilon, non_decrease_iter, relative)
            for _ in range(num_models)
        ]

    @property
    def stop(self):
        """Evaluate the model."""
        count = sum([es.stop for es in self.model_ensemble_early_stopping])
        return count >= self.fraction

    def reset(self, hard=True):
        """Reset moving average and minimum values.

        Parameters
        ----------
        hard: bool, optional (default=True).
            If true, reset moving average and min_value.
            If false, reset only moving average.
        """
        for es_algorithm in self.model_ensemble_early_stopping:
            es_algorithm.reset(hard=hard)

    def _evaluate_model(self, i, state):
        self.dynamical_model.set_head(i)
        self.reward_model.set_head(i)
        observation = self.simulate(state, self.policy, stack_obs=True)
        return observation.reward.sum(-1).mean()

    def update(self, state):
        """Update estimation."""
        if self.eval_frequency > 0 and (self.step_counts + 1) % self.eval_frequency > 0:
            self.step_counts += 1
            return

        self.step_counts = 0
        with torch.no_grad(), PredictionStrategy(
            self.dynamical_model, self.reward_model, prediction_strategy="set_head"
        ):
            estimated_returns = run_parallel_returns(
                self._evaluate_model, [(i, state) for i in range(self.num_models)]
            )

        for i in range(self.num_models):
            self.model_ensemble_early_stopping[i].update(estimated_returns[i])
