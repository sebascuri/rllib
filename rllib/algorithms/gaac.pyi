from typing import Any

from .ac import ActorCritic
from .gae import GAE

class GAAC(ActorCritic):
    gae: GAE
    def __init__(self, lambda_: float, *args: Any, **kwargs: Any) -> None: ...
