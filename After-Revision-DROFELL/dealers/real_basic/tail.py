from operators.container.container import Container
from operators.core.tail.real import ModelTail
from operators.core.sampler.real import (
    batch_sample,
    batch_sample_greedy_no_logprobs,
)
from ..common.brain import *
from .decode_graph import HiddenInput, TailDecodeGraphRunner


def uses_greedy_no_logprobs(
    sampling_params: List[SamplingParams],
) -> bool:
    return all(
        params.temperature < 1e-5
        and params.logprobs is None
        and params.presence_penalty == 0.0
        and params.frequency_penalty == 0.0
        and params.repetition_penalty == 1.0
        for params in sampling_params
    )


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


async def decode_hidden_states_to_sampling_outputs(
    model_tail: ModelTail,
    container: Container,
    hidden_states: HiddenInput,
    tasks: List[DecodeTailTask],
) -> List[Tuple[int, Dict[int, float]]]:
    if len(hidden_states) != len(tasks):
        raise ValueError("hidden state and decode tail task counts differ")
    for task in tasks:
        if task.begin_metric is not None:
            task.begin_metric()
    if isinstance(hidden_states, torch.Tensor):
        stacked_hidden_states = hidden_states
    else:
        stacked_hidden_states = await container.step(
            "stack_fused_decode_tail", torch.stack, tensors=hidden_states
        )
    logits = await model_tail.to_logits(
        container, hidden_states=stacked_hidden_states
    )
    sampling_params = [task.sampling_params for task in tasks]
    use_greedy_fast_path = uses_greedy_no_logprobs(sampling_params)
    if use_greedy_fast_path:
        next_tokens = await batch_sample_greedy_no_logprobs(container, logits)
        log_probs = [{} for _ in tasks]
    else:
        next_tokens, log_probs = await batch_sample(
            container=container,
            seqs_logits=logits,
            sampling_params=sampling_params,
            output_token_ids=[task.seq_core.output_token_ids for task in tasks],
            prompt_token_ids=[task.seq_core.prompt_token_ids for task in tasks],
        )
    outputs: List[Tuple[int, Dict[int, float]]] = []
    for task, next_token, task_log_probs in zip(tasks, next_tokens, log_probs):
        task.seq_core.append_token_id(next_token)
        outputs.append((next_token, task_log_probs))
        if task.finish_metric is not None:
            task.finish_metric()
    return outputs


async def hidden_states_to_sampling_output(
    model_tail: ModelTail,
    container: Container,
    operates: List[OperateHiddenStateToSamplingOutput],
    decode_graph_runner: TailDecodeGraphRunner,
    decode_graph_binding: str,
) -> None:
    is_evil_fast = check_evil_fast(operates)
    graph_hidden_states: HiddenInput
    if is_evil_fast:
        hidden_states, aux = operates[0].carrier.tensor[0]
        graph_hidden_states = hidden_states
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
        graph_hidden_states = input_hidden_states

    sampling_params = [o.sampling_params for o in operates]
    use_greedy_fast_path = uses_greedy_no_logprobs(sampling_params)
    is_decode = all(not operate.is_prefill for operate in operates)
    graph_fallback = decode_graph_runner.fallback_reason(
        batch_size=len(operates),
        use_greedy_fast_path=use_greedy_fast_path,
    )
    use_graph = is_decode and graph_fallback is None
    if is_decode and graph_fallback is not None:
        decode_graph_runner.record_fallback(graph_fallback)

    if use_graph:
        next_token = await decode_graph_runner.run(
            key=(decode_graph_binding, len(operates)),
            model_tail=model_tail,
            hidden_states=graph_hidden_states,
            container=container,
        )
        log_probs = [{} for _ in operates]
    else:
        if is_evil_fast:
            hidden_states = hidden_states.squeeze(1)
        else:
            hidden_states = await container.step(
                "stack", torch.stack, tensors=graph_hidden_states
            )
        logits = await model_tail.to_logits(
            container, hidden_states=hidden_states
        )
        if use_greedy_fast_path:
            next_token = await batch_sample_greedy_no_logprobs(
                container, logits
            )
            log_probs = [{} for _ in operates]
        else:
            next_token, log_probs = await batch_sample(
                container=container,
                seqs_logits=logits,
                sampling_params=sampling_params,
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
