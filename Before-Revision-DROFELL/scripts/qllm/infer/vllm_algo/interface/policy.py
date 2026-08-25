from typing import List
from .sequence import SequenceGroup
from .basic_types import *


class Policy:

    def get_priority(
            self,
            now: Time,
            seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
            self,
            now: Time,
            seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )
