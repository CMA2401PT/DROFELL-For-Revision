from dataclasses import dataclass
from typing import List

from ..interface.block import BlockTable
from .basic_types import VirtualBlockNumber

PtuningBlocks = List[VirtualBlockNumber]


@dataclass
class PtuningHandler:
    ptuning_id: str
    ptuning_blocks: PtuningBlocks
    build_prompt_str: str
    ptuning_table: BlockTable  # for free
    ref_count: int = 0
