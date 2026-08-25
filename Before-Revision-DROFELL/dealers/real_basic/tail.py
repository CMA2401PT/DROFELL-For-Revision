from operators.container.container import Container
from operators.core.tail.real import ModelTail
from operators.core.sampler.real import batch_sample
from ..common.brain import *


def check_evil_fast(operates: List[OperateHiddenStateToSamplingOutput]):
    start = -1
    curr = -1
    for op in operates:
        t = op.carrier.tensor
        if not isinstance(op.carrier.tensor, tuple):
            return False
        _, _, _start, _num, _this = t
        if _num != len(operates):
            return False
        if start == -1:
            start = _start
            curr = _start
        elif start != _start:
            return False
        if _this != curr:
            return False
        else:
            curr += 1
    return True


async def hidden_states_to_sampling_output(
    model_tail: ModelTail,
    container: Container,
    operates: List[OperateHiddenStateToSamplingOutput],
) -> None:
    is_evil_fast = check_evil_fast(operates)
    if is_evil_fast:
        hidden_states, _ = operates[0].carrier.tensor[0]
        hidden_states.squeeze_(1)
    else:
        input_hidden_states: List[torch.Tensor] = []
        for o in operates:
            if o.cpu_in:
                input_hidden_states.append(await container.data_cpu_to_device(o.carrier.tensor))
            else:
                t = o.carrier.tensor
                if isinstance(o.carrier.tensor, tuple):
                    t = t[1]()
                input_hidden_states.append(t)
        hidden_states = await container.step("stack", torch.stack, tensors=input_hidden_states)

    logits = await model_tail.to_logits(container, hidden_states=hidden_states)
    next_token, log_probs = await batch_sample(
        container=container,
        seqs_logits=logits,
        sampling_params=[o.sampling_params for o in operates],
        output_token_ids=[o.seq_core.output_token_ids for o in operates],
        prompt_token_ids=[o.seq_core.prompt_token_ids for o in operates])
    for i, o in enumerate(operates):
        o.carrier.free()
        o.seq_core.append_token_id(next_token[i])
        o.fut.set_result((next_token[i], log_probs[i]))


async def compute_logits_loss_and_grad(
    model_tail: ModelTail,
    container: Container,
    operate: OperateComputeLogitsLossAndGrad,
) -> None:
    hidden_states = operate.carrier.tensor
    label_ids = await container.data_cpu_to_device(operate.label_ids)
    if operate.cpu_in:
        hidden_states = await container.data_cpu_to_device(hidden_states)
    float_loss, grad_data = await model_tail.compute_loss_and_grad(container, operate.criterion, hidden_states, label_ids)
    sp = None
    if operate.cpu_in:
        grad_data = await container.data_device_to_cpu(grad_data, wait=False)
        sp = container.sync_point()
    operate.set_result((float_loss, DeviceTensorCarrier(grad_data, operate.carrier.no_gc)), sp)
    operate.carrier.free()
