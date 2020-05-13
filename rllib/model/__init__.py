from .abstract_model import AbstractModel
from .linear_model import LinearModel
from .nn_model import NNModel
from .gp_model import ExactGPModel, SparseGPModel, RandomFeatureGPModel
from .ensemble_model import EnsembleModel
from .derived_model import TransformedModel, OptimisticModel, ExpectedModel

"""
A model is a mapping from states and actions to next_state distributions. 

It should will accept raw states and actions and, if needed, perform any transformation
inside the model.

The derived models provide such functionality. 
"""
