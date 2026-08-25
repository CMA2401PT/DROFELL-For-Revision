from typing import Dict, List
import enum

TokenID = int
Text = str
LogProb = float
SampleLogProb = Dict[TokenID, LogProb]
SeqSampleLogprobsInSeqGroup = List[SampleLogProb]

VirtualBlockNumber = int
VirtualSlotNumber = int


class VirtualDevice(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


SeqID = str
SeqGroupSelectedSeqIDs = List[SeqID]
TokenDecoded = str
Time = float
