"""Simplest TD-Learning algorithm."""

from rllib.util.losses import EntropyLoss, KLLoss

from .abstract_algorithm import AbstractAlgorithm


class FittedValueEvaluationAlgorithm(AbstractAlgorithm):
    """Fitted Value Evaluation Algorithm.

    References
    ----------
    Munos, R., & Szepesvári, C. (2008).
    Finite-Time Bounds for Fitted Value Iteration. JMLR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.entropy_loss = EntropyLoss()
        self.kl_loss = KLLoss()
        self.pathwise_loss = None
