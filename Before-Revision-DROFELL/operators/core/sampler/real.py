from typing import Dict, List,  Tuple
import torch

from ...container import Container
from .common import SamplingParams, SamplingType
_SAMPLING_EPS = 1e-5


async def batch_sample(
    container: Container,
    seqs_logits: torch.Tensor,
    sampling_params: List[SamplingParams],
    output_token_ids: List[List[int]],
    prompt_token_ids: List[List[int]],
) -> Tuple[List[int], List[Dict[int, float]]]:
    probs, logprobs = await container.step(
        "batch_sample_device", _batch_sample_device_stage,
        seqs_logits=seqs_logits,
        sampling_params=sampling_params,
        output_token_ids=output_token_ids,
        prompt_token_ids=prompt_token_ids,
    )

    sample_results = await _sample(
        probs, logprobs, [p.sampling_type for p in sampling_params],
        container
    )
    log_probs = await _batch_get_logprobs(
        sample_results=sample_results,
        logprobs=logprobs,
        num_logprobs_outs=[(0 if p.logprobs is None else p.logprobs) for p in sampling_params],
        container=container
    )
    return sample_results, log_probs


def _batch_sample_device_stage(
    seqs_logits: torch.Tensor,
    sampling_params: List[SamplingParams],
    output_token_ids: List[List[int]],
    prompt_token_ids: List[List[int]],
):
    vocab_size = seqs_logits.size(-1)

    seqs_logits = _batch_apply_penalties(
        seqs_logits,
        [p.presence_penalty for p in sampling_params],
        [p.frequency_penalty for p in sampling_params],
        [p.repetition_penalty for p in sampling_params],
        output_token_ids,
        prompt_token_ids
    )

    temperatures = [(p.temperature if p.temperature >= _SAMPLING_EPS else 1.0) for p in sampling_params]
    if any(t != 1.0 for t in temperatures):
        t = torch.tensor(temperatures, dtype=seqs_logits.dtype, device=seqs_logits.device)
        # Use in-place division to avoid creating a new tensor.
        seqs_logits.div_(t.unsqueeze(dim=1))

    # Apply top-p and top-k truncation.
    top_ps: List[float] = [p.top_p for p in sampling_params]
    min_ps: List[float] = [p.min_p for p in sampling_params]
    top_ks: List[int] = [(vocab_size if p.top_k == -1 else min(p.top_k, vocab_size)) for p in sampling_params]
    do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
    do_top_k = any(k != vocab_size for k in top_ks)
    if do_top_p or do_top_k:
        seqs_logits = _batch_apply_top_p_top_k(seqs_logits, top_ps, top_ks)

    do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
    if do_min_p:
        seqs_logits = _batch_apply_min_p(seqs_logits, min_ps)

    probs = torch.softmax(seqs_logits, dim=-1, dtype=torch.float)
    logprobs = torch.log_softmax(seqs_logits, dim=-1, dtype=torch.float)
    return probs, logprobs


def _batch_apply_penalties(
        logits: torch.Tensor,
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        output_token_ids: List[List[int]],
        prompt_token_ids: List[List[int]],
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    for i in range(num_seqs):
        p = presence_penalties[i]
        f = frequency_penalties[i]
        r = repetition_penalties[i]
        if abs(p) < _SAMPLING_EPS and abs(f) < _SAMPLING_EPS and abs(
                r - 1.0) < _SAMPLING_EPS:
            continue
        break
    else:
        # Return early if all sequences have zero penalties.
        return logits

    _, prompt_mask = _batch_get_bin_counts_and_mask(logits, prompt_token_ids, vocab_size, num_seqs)
    output_bin_counts, output_mask = _batch_get_bin_counts_and_mask(logits, output_token_ids, vocab_size, num_seqs)

    repetition_penalties_tenser = torch.tensor(repetition_penalties, dtype=logits.dtype, device=logits.device)
    frequency_penalties_tenser = torch.tensor(frequency_penalties, dtype=logits.dtype, device=logits.device)
    presence_penalties_tenser = torch.tensor(presence_penalties, dtype=logits.dtype, device=logits.device)

    repetition_penalties_tenser = repetition_penalties_tenser[:, None].repeat(1, vocab_size)
    repetition_penalties_tenser[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties_tenser, logits * repetition_penalties_tenser)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties_tenser.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties_tenser.unsqueeze(dim=1) * output_mask
    return logits


def _batch_get_bin_counts_and_mask(
        logits: torch.Tensor,
        tokens: List[List[int]],
        vocab_size: int,
        num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(tokens) for tokens in tokens)
    padded_tokens = [
        tokens + [vocab_size] * (max_len - len(tokens)) for tokens in tokens
    ]
    tokens_tensor = torch.tensor(padded_tokens,
                                 dtype=torch.long,
                                 device=logits.device)

    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=logits.device)
    bin_counts.scatter_add_(1, tokens_tensor, torch.ones_like(tokens_tensor))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def _batch_apply_top_p_top_k(
        logits: torch.Tensor,
        top_ps: List[float],
        top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))
    return logits


def _batch_apply_min_p(
        logits: torch.Tensor,
        min_ps: List[float],
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    min_p = torch.tensor(min_ps, dtype=logits.dtype, device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill(tokens_to_remove, -float("inf"))

    return logits


async def _batch_get_logprobs(
    sample_results: List[int],
    logprobs: torch.Tensor,
    num_logprobs_outs: List[int],  # if None then 0
    container: Container,
) -> List[Dict[int, float]]:
    # Batched query for logprobs of selected token
    batched_logprobs_query_result: torch.Tensor = logprobs[[
        [i for i in range(len(sample_results))],
        sample_results
    ]]

    # Batched query for logprobs of topk tokens
    largest_num_logprobs = max(num_logprobs_outs)
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = await container.step(
            "topk", torch.topk,
            # Batched query for logprobs of topk tokens
            input=logprobs, k=largest_num_logprobs, dim=-1)
        top_logprobs = await container.data_device_to_cpu(top_logprobs, wait=False)
        top_token_ids = await container.data_device_to_cpu(top_token_ids, wait=True)
    else:
        top_logprobs, top_token_ids = None, None

    # Gather results
    result_sample_logprobs: List[Dict[int, float]] = []
    for seq_i, (num_logprobs_out, sample_result) in enumerate(zip(num_logprobs_outs, sample_results)):
        result_sample_logprob: Dict[int, float] = {
            sample_result: batched_logprobs_query_result[seq_i].item()
        }
        if top_logprobs is not None and top_token_ids is not None:
            result_sample_logprob.update(
                zip(
                    top_token_ids[seq_i, :num_logprobs_out].tolist(),  # type:ignore
                    top_logprobs[seq_i, :num_logprobs_out].tolist()  # type:ignore
                )
            )
        result_sample_logprobs.append(result_sample_logprob)
    return result_sample_logprobs


async def _sample(
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampling_types: List[SamplingType],
        container: Container,
) -> List[int]:
    categorized_seq_ids: Dict[SamplingType, List[int]] = {t: [] for t in SamplingType}
    for seq_group_i, sampling_type in enumerate(sampling_types):
        categorized_seq_ids[sampling_type].append(seq_group_i)

    sample_results_dict: Dict[int, int] = {}
    for sampling_type in SamplingType:
        seq_ids = categorized_seq_ids[sampling_type]
        num_tokens = len(seq_ids)
        if num_tokens == 0:
            continue
        if sampling_type == SamplingType.GREEDY:
            category_logprobs = logprobs[seq_ids]
            sample_results = torch.argmax(category_logprobs, dim=-1)
        elif sampling_type == SamplingType.RANDOM:
            category_probs = probs[seq_ids]
            sample_results = torch.multinomial(category_probs, num_samples=1, replacement=True)
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")
        sample_results = (await container.data_device_to_cpu(sample_results, wait=True)).tolist()
        sample_results_dict.update(zip(seq_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_types))
    ]
    return sample_results
