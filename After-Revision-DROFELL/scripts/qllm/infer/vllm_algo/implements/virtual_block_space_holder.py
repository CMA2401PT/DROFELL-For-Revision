from ..interface.basic_types import VirtualDevice
from ..interface.block import BlockTable, VirtualPhysicalTokenBlock
from ..interface.block_space import VirtualBlockSpaceHolder


class VllmLikeVirtualBlockSpaceHolder(VirtualBlockSpaceHolder):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.

    represents whether a block is still available
    """

    def __init__(
            self,
            device: VirtualDevice,
            block_size: int,
            num_blocks: int,
    ) -> None:
        self.device: VirtualDevice = device
        self._block_size: int = block_size
        self.num_blocks: int = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            block = VirtualPhysicalTokenBlock(device=device, block_number=i, block_size=block_size)
            self.free_blocks.append(block)

    def reset(self):
        for i in range(self.num_blocks):
            block = VirtualPhysicalTokenBlock(device=self.device, block_number=i, block_size=self._block_size)
            self.free_blocks.append(block)

    @property
    def block_size(self) -> int:
        return self._block_size

    def allocate(self) -> VirtualPhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block: VirtualPhysicalTokenBlock = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: VirtualPhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_total_blocks(self) -> int:
        return self.num_blocks
