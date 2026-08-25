from typing import Any, Dict
from ..interface.basic_types import Time
from ..interface.sequence import SequenceGroup
from ..interface.policy import Policy


class FCFS(Policy):

    def get_priority(
            self,
            now: Time,
            seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs: Dict[str, Any]) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
