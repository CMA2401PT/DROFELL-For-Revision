from operators.core.head.real import ModelHead
from operators.container.container import Container
from ..common.brain import *
from . import flags
_CPU = torch.device("cpu")
_evil_fast_alloc = 0


async def infer_tensor_to_embedding_tokens(
    head: ModelHead,
    container: Container,
    operates: List[OperateInferTensorToEmbeddingTokens]
) -> None:
    ranges: List[Tuple[int, int]] = []
    offset = 0
    merged_tokens: List[int] = []
    for operate in operates:
        t = operate.tokens
        l = len(t)
        merged_tokens += t
        ranges.append((offset, offset + l))
        offset += l

    input_tokens_tensor = torch.tensor(merged_tokens, dtype=torch.int32, device=_CPU)
    input_tokens_tensor = await container.data_cpu_to_device(input_tokens_tensor)
    hidden_states = await head.infer_to_embedding_tokens(container=container, input_tokens_tensor=input_tokens_tensor, dtype=torch.float16)

    # check evil fast
    if flags.ALLOW_2WAY_DISPATCH and all((not op.is_prefill) for op in operates) and all((not op.cpu_out) for op in operates):
        # can evil fast
        hidden_states = await container.step("unsqueeze", torch.unsqueeze, input=hidden_states, dim=1)
        direct_result = (hidden_states, None)
        start = _evil_fast_alloc
        num = len(operates)

        def make_evil_result(i: int):
            global _evil_fast_alloc

            def lazy():
                return hidden_states[i].view(-1)
            this = _evil_fast_alloc
            _evil_fast_alloc += 1
            return (direct_result if i == 0 else None), lazy, start, num, this
        for i, o in enumerate(operates):
            o.set_result(DeviceTensorCarrier(make_evil_result(i), None))  # type:ignore
    else:
        outs: List[torch.Tensor] = []
        for i, o in enumerate(operates):
            t = hidden_states[ranges[i][0]:ranges[i][1]].clone()
            if not o.is_prefill:
                t = t.view(-1)
            outs.append(await container.data_device_to_cpu(t, wait=False) if o.cpu_out else t)
        sp = container.sync_point() if any(o.cpu_out for o in operates) else None
        for i, o in enumerate(operates):
            o.set_result(DeviceTensorCarrier(outs[i], None), sp if o.cpu_out else None)


async def train_tensor_to_embedding_tokens(
    head: ModelHead,
    container: Container,
    operate: OperateTrainTensorToEmbeddingTokens
) -> None:
    hidden_states = operate.hidden_states
    assert hidden_states.device != container.device
    input_tokens_tensor = await container.data_cpu_to_device(hidden_states)
    hidden_states = await head.train_to_embedding_tokens(container, input_tokens_tensor, operate.dtype)
    sp = None
    if operate.cpu_out:
        hidden_states = await container.data_device_to_cpu(hidden_states, wait=False)
        sp = container.sync_point()
    operate.set_result(DeviceTensorCarrier(hidden_states, None), sp)
