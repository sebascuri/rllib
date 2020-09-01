from typing import Any

from rllib.agent.on_policy.on_policy_agent import OnPolicyAgent
from rllib.dataset.experience_replay import ExperienceReplay

from .off_policy_agent import OffPolicyAgent

class ISERAgent(OffPolicyAgent):
    def __init__(
        self,
        base_agent: OnPolicyAgent,
        memory: ExperienceReplay,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
