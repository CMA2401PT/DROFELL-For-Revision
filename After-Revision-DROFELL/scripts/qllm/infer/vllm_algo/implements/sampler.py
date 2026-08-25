from typing import Tuple
import torch
from ..interface.basic_types import SeqGroupSelectedSeqIDs
from ..interface.sample import Sampler, SamplingMetadata, SequenceSampleOutput
from ..interface.basic_types import *
from ..interface.sampling_params import SamplingParams, SamplingType
from ..interface.sequence import SeqSampleLogprobsInSeqGroup

ParentSeqIDOffsetInSeqGroupIDs = int
_SAMPLING_EPS = 1e-5


class RealWorldSampler(Sampler):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> List[List[SequenceSampleOutput]]:
        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)
        # Apply presence and frequency penalties.
        presence_penalties, frequency_penalties, repetition_penalties = (_get_penalties(sampling_metadata))
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        assert len(repetition_penalties) == logits.shape[0]
        logits = _apply_penalties(logits,
                                  sampling_metadata,
                                  presence_penalties, frequency_penalties,
                                  repetition_penalties)

        # Apply temperature scaling.
        temperatures = _get_temperatures(sampling_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks, min_ps = _get_top_p_top_k_min_p(
            sampling_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
        if do_min_p:
            logits = _apply_min_p(logits, min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata, sample_logprobs)


def _get_penalties(
        sampling_metadata: SamplingMetadata
) -> Tuple[List[float], List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    for seq_group_sampling_params in sampling_metadata.seq_groups_sampling_params:
        seq_group_seq_ids, sampling_params = seq_group_sampling_params
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        r = sampling_params.repetition_penalty
        presence_penalties += [p] * len(seq_group_seq_ids)
        frequency_penalties += [f] * len(seq_group_seq_ids)
        repetition_penalties += [r] * len(seq_group_seq_ids)
    return presence_penalties, frequency_penalties, repetition_penalties


def _get_bin_counts_and_mask(
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


def _apply_logits_processors(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    logits_row_idx = 0
    found_logits_processors = False
    for seq_ids, _ in sampling_metadata.seq_groups_sampling_params:
        # logits_processors = seq_group_sampling_params.logits_processors
        # if logits_processors:
        #     found_logits_processors = True
        #     for seq_id in seq_ids:
        #         logits_row = logits[logits_row_idx]
        #         token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
        #         for logits_processor in logits_processors:
        #             logits_row = logits_processor(token_ids, logits_row)
        #         logits[logits_row_idx] = logits_row
        #         logits_row_idx += 1
        # else:
        logits_row_idx += len(seq_ids)
    if found_logits_processors:
        assert logits_row_idx == logits.shape[0]
    return logits


def _apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
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

    prompt_tokens, output_tokens = (_get_prompt_and_output_tokens(sampling_metadata))
    assert len(prompt_tokens) == logits.shape[0]
    assert len(output_tokens) == logits.shape[0]

    _, prompt_mask = _get_bin_counts_and_mask(logits, prompt_tokens, vocab_size, num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(logits, output_tokens, vocab_size, num_seqs)

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


def _get_prompt_and_output_tokens(
        sampling_metadata: SamplingMetadata,
) -> Tuple[List[List[int]], List[List[int]]]:
    prompt_tokens: List[List[int]] = []
    output_tokens: List[List[int]] = []
    for seq_group_sampling_params in sampling_metadata.seq_groups_sampling_params:
        seq_ids, _ = seq_group_sampling_params
        for seq_id in seq_ids:
            seq_data = sampling_metadata.seq_data[seq_id]
            prompt_tokens.append(seq_data.prompt_token_ids)
            output_tokens.append(seq_data.output_token_ids)
    return prompt_tokens, output_tokens


def _get_temperatures(sampling_metadata: SamplingMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for seq_group_sampling_params in sampling_metadata.seq_groups_sampling_params:
        seq_group_seq_ids, sampling_params = seq_group_sampling_params
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        temperatures += [temperature] * len(seq_group_seq_ids)
    return temperatures


def _get_top_p_top_k_min_p(
        sampling_metadata: SamplingMetadata,
        vocab_size: int,
) -> Tuple[List[float], List[int], List[float]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    min_ps: List[float] = []
    for seq_group_sampling_params in sampling_metadata.seq_groups_sampling_params:
        seq_group_seq_ids, sampling_params = seq_group_sampling_params
        top_p = sampling_params.top_p
        min_p = sampling_params.min_p
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        top_ps += [top_p] * len(seq_group_seq_ids)
        top_ks += [top_k] * len(seq_group_seq_ids)
        min_ps += [min_p] * len(seq_group_seq_ids)
    return top_ps, top_ks, min_ps


def _apply_top_p_top_k(
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


def _apply_min_p(
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


def _greedy_sample(
        selected_seq_groups: List[Tuple[SeqGroupSelectedSeqIDs, SamplingParams]],
        logprobs: torch.Tensor,
) -> List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]]:
    samples = torch.argmax(logprobs, dim=-1).cpu()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, _ = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, "Greedy sampling should have only one seq."
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx].item()]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results


def _random_sample(
        selected_seq_groups: List[Tuple[SeqGroupSelectedSeqIDs, SamplingParams]],
        is_prompts: List[bool],
        probs: torch.Tensor,
) -> List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]]:
    # Find the maximum best_of value of the prompt phase requests.
    max_best_of = 1
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        if is_prompt:
            seq_ids, _ = seq_group
            max_best_of = max(max_best_of, 1)  # sampling_params.best_of)
    random_samples = torch.multinomial(probs,
                                       num_samples=max_best_of,
                                       replacement=True).cpu()
    sample_idx = 0
    results: List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]] = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, _ = seq_group
        num_parent_seqs: int = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * 1  # sampling_params.best_of
            next_token_ids: List[TokenID] = random_samples[sample_idx, :1].tolist()  # sampling_params.best_of
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids: List[TokenID] = random_samples[sample_idx:sample_idx + num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == probs.size(0)
    return results


# def _beam_search_sample(
#         selected_seq_groups: List[Tuple[SeqGroupSelectedSeqIDs, SamplingParams]],
#         is_prompts: List[bool],
#         seq_data: Dict[SeqID, SequenceTokenIDsData],
#         logprobs: torch.Tensor,
# ) -> List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]]:
#     # We sample 2 * beam_width candidates to make sure that with high
#     # probability we can get `beam_width` candidates in addition to
#     # the finished sequences for the next iteration. See
#     # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
#     # for details. See also HF reference:
#     # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
#     #
#     # NOTE: Beam search is not vectorized, so its speed can be slower than
#     # other sampling methods.
#     sample_idx = 0
#     results = []
#     for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
#         seq_ids, sampling_params = seq_group
#         num_parent_seqs = len(seq_ids)
#         beam_width = sampling_params.best_of
#         seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
#         if is_prompt:
#             # Prompt phase.
#             assert num_parent_seqs == 1, (
#                 "Prompt input should have only one seq.")
#             parent_ids = [0] * (2 * beam_width)
#             _, next_token_ids = torch.topk(seq_group_logprobs[0],
#                                            2 * beam_width)
#             next_token_ids = next_token_ids.tolist()
#         else:
#             # Generation phase.
#             cumulative_logprobs = [
#                 seq_data[seq_id].cumulative_logprob for seq_id in seq_ids
#             ]
#             cumulative_logprobs = torch.tensor(
#                 cumulative_logprobs,
#                 dtype=torch.float,
#                 device=seq_group_logprobs.device)
#             seq_group_logprobs = (seq_group_logprobs +
#                                   cumulative_logprobs.unsqueeze(dim=1))
#             _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
#                                      2 * beam_width)
#             topk_ids = topk_ids.tolist()
#             vocab_size = seq_group_logprobs.size(-1)
#             parent_ids = [i // vocab_size for i in topk_ids]
#             next_token_ids = [i % vocab_size for i in topk_ids]
#         results.append((next_token_ids, parent_ids))
#         sample_idx += num_parent_seqs
#     assert sample_idx == logprobs.size(0)
#     return results


def _sample(
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
) -> List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]]:
    categorized_seq_group_ids: Dict[SamplingType, List[int]] = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for seq_group_i, seq_group_sampling_params in enumerate(sampling_metadata.seq_groups_sampling_params):
        _, sampling_params = seq_group_sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(seq_group_i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    for sampling_type in SamplingType:
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups_seq_ids_and_sampling_params = [sampling_metadata.seq_groups_sampling_params[i] for i in
                                                  seq_group_ids]
        is_prompts = sampling_metadata.is_prompts
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        if sampling_type == SamplingType.GREEDY:
            category_logprobs = logprobs[sample_indices]
            sample_results = _greedy_sample(
                seq_groups_seq_ids_and_sampling_params, category_logprobs)
        elif sampling_type == SamplingType.RANDOM:
            category_probs = probs[sample_indices]
            sample_results = _random_sample(
                seq_groups_seq_ids_and_sampling_params, is_prompts, category_probs)
        # elif sampling_type == SamplingType.BEAM:
        #     category_logprobs = logprobs[sample_indices]
        #     seq_datas = sampling_metadata.seq_data
        #     sample_results = _beam_search_sample(seq_groups_seq_ids_and_sampling_params,
        #                                          is_prompts,
        #                                          seq_datas,
        #                                          category_logprobs
        #                                          )
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups_sampling_params))
    ]
    return sample_results


def _get_logprobs(
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        sample_results: List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]],
) -> List[SeqSampleLogprobsInSeqGroup]:
    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
    sample_idx = 0
    for (seq_group_sampling_params, sample_result) in zip(sampling_metadata.seq_groups_sampling_params, sample_results):
        seq_ids, sampling_params = seq_group_sampling_params
        next_token_ids, parent_id_offsets = sample_result
        num_parent_seqs = len(seq_ids)
        batched_logprobs_query_seq_indices.extend(
            [sample_idx + parent_id_offset for parent_id_offset in parent_id_offsets]
        )
        batched_logprobs_query_token_indices.extend(next_token_ids)
        if sampling_params.logprobs is not None:
            largest_num_logprobs = max(
                largest_num_logprobs, sampling_params.logprobs)
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)

    # Batched query for logprobs of selected token
    batched_logprobs_query_result = logprobs[[
        batched_logprobs_query_seq_indices,
        batched_logprobs_query_token_indices
    ]].cpu()

    # Batched query for logprobs of topk tokens
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(
            logprobs, largest_num_logprobs, dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = None, None

    # Gather results
    result_sample_logprobs: List[SeqSampleLogprobsInSeqGroup] = []
    sample_idx = 0
    query_result_idx = 0
    for seq_group_sampling_params, sample_result in zip(sampling_metadata.seq_groups_sampling_params, sample_results):
        seq_ids, sampling_params = seq_group_sampling_params
        next_token_ids, parent_ids_offset = sample_result

        # Sample logprobs
        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SeqSampleLogprobsInSeqGroup = []
        for next_token_id, parent_id_offset in zip(next_token_ids, parent_ids_offset):
            sample_logprobs_dict: dict[TokenID, LogProb] = {
                next_token_id: batched_logprobs_query_result[query_result_idx].item(
                )
            }
            query_result_idx += 1
            if num_logprobs > 0:
                assert largest_num_logprobs > 0 and sampling_params.logprobs is not None and top_token_ids is not None and top_logprobs is not None
                sample_logprobs_dict.update(
                    zip(  # type:ignore
                        top_token_ids[sample_idx + parent_id_offset, :num_logprobs].tolist(),  # type:ignore
                        top_logprobs[sample_idx + parent_id_offset, :num_logprobs].tolist()  # type:ignore
                    )
                )
            group_sample_logprobs.append(sample_logprobs_dict)
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)

    return result_sample_logprobs


def _build_sampler_output(
        sample_results: List[Tuple[List[TokenID], List[ParentSeqIDOffsetInSeqGroupIDs]]],
        sampling_metadata: SamplingMetadata,
        sample_logprobs: List[SeqSampleLogprobsInSeqGroup],
) -> List[List[SequenceSampleOutput]]:
    sampler_output: List[List[SequenceSampleOutput]] = []
    seq_groups_sampling_params = sampling_metadata.seq_groups_sampling_params
    for (seq_groups_sampling_param, sample_result, group_sample_logprobs) in \
            zip(seq_groups_sampling_params, sample_results, sample_logprobs):
        seq_groups_sampling_param: Tuple[SeqGroupSelectedSeqIDs,
                                         SamplingParams]
        sample_result: Tuple[List[TokenID],
                             List[ParentSeqIDOffsetInSeqGroupIDs]]
        seq_ids, _ = seq_groups_sampling_param
        seq_ids: SeqGroupSelectedSeqIDs
        next_token_ids, parent_ids = sample_result
        seq_outputs: List[SequenceSampleOutput] = []
        for parent_id, next_token_id, logprobs in zip(parent_ids, next_token_ids, group_sample_logprobs):
            seq_outputs.append(SequenceSampleOutput(seq_ids[parent_id], next_token_id, logprobs))
        sampler_output.append(seq_outputs)
    return sampler_output
