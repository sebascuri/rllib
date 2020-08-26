"""Utilities for models."""


class PredictionStrategy(object):
    """Context manager to disable optimization steps temporarily.

    Gradients and momentum terms will be zero-ed.

    Parameters
    ----------
    models: AbstractModel.
        Model to use.
    prediction_strategy: str.
        Prediction strategy to set.
    """

    def __init__(self, *models, prediction_strategy="moment_matching"):
        self.models = models
        self.prediction_strategy = prediction_strategy

        self._prediction_strategies = []
        self._heads = []
        for model in self.models:
            self._prediction_strategies.append(model.get_prediction_strategy())
            self._heads.append(model.get_head())

    def __enter__(self):
        """Freeze the parameters."""
        for model in self.models:
            model.set_prediction_strategy(self.prediction_strategy)

    def __exit__(self, *args):
        """Unfreeze the parameters."""
        for model, prediction_strategy, head in zip(
            self.models, self._prediction_strategies, self._heads
        ):
            model.set_prediction_strategy(prediction_strategy)
            model.set_head(head)
