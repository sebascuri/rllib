from typing import Any

from .abstract_policy import AbstractPolicy

class SetDeterministic(object):
    policy: AbstractPolicy
    cache: bool
    def __init__(self, policy: AbstractPolicy) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
