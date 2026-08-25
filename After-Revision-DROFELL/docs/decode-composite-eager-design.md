# Composite Decode Head/Tail Execution and CUDA Graphs

## 1. Scope

This document explains how decode delegates an adjacent head, one decoder group, and a tail to one decoder-generating operation, how exact matches use a fused CUDA graph, and how other cases execute in stages.

The design explicitly excludes:

- prefill, which continues using independent head, decoder, and tail operations;
- merging across decoder groups, each of which retains its own operation and transfer boundary;
- cross-Brain delegation, because composition requires adjacent shims to share one Brain.

## 2. Layer responsibilities

```text
InferNode
  interprets Navigate, checks adjacency/same-Brain ownership,
  waits for shim-ordering conditions, and advances the pointer
    |
    v
DecodersInferBindingShim.generating
  marks participating shim controls as running, creates pure task
  descriptions, and submits one parent operation
    |
    v
Brain.operates_do_generating
  the only mailbox; selects and batches by the underlying decoder Binding
    |
    v
MemoryBrain / SwapperCore
  models, trials, rolls back, or commits the complete composite operation
    |
    v
run_generating
  group by head Binding -> one decoder batch -> group by tail Binding
```

`InferNode` neither computes head/tail tensors nor mutates Brain queues directly. Head and tail do not create nested `Operate` objects; they become `DecodeHeadTask` / `DecodeTailTask` components of the decoder parent operation. The Brain still schedules one generating-mailbox item.

## 3. Fusion plan

`_DecodeFusionPlan` describes one contiguous local decode segment in the current `Navigate`:

| Current position | Delegated form | Navigation span |
|---|---|---:|
| head | head + decoder | 2 |
| head | head + decoder + tail | 3 |
| decoder | decoder + tail | 2 |
| decoder | decoder | 1, ordinary path |

Composition requires:

1. a decode task;
2. strict model adjacency in `Navigate`;
3. initialized target shims in the current `InferNode` cache;
4. identical `brain` ownership for the head/tail and decoder shims.

If the next target is still initializing, resides on another node or Brain, or has the wrong model type, the current step uses the standalone path. This fallback does not change navigation or carrier semantics.

## 4. Shim-control lifecycle

Before composition, `InferNode` calls `can_continue_or_reoder()` on each participating shim. This waits for the step frontier but neither creates a task nor takes execution ownership. Once every participant can run, `DecodersInferBindingShim.generating()`:

1. calls `start_decode()` on head/tail shims to validate the step, set `running`, and create task descriptions;
2. marks the decoder shim as running and updates slot/block callbacks;
3. creates one `OperateDoGenerating` and appends it to the decoder mailbox;
4. waits for the parent operation's single result;
5. completes the current step for decoder, head, and tail controls together.

If the operation returns a carrier, `DecodeGeneratingResult.carrier` is non-empty. If an embedded tail sampled a token, `sampling_output` is non-empty. Exactly one of the two must exist.

This ownership rule prevents `InferNode` from mutating control state independently and prevents the Brain from creating nested head/tail operations, either of which would reintroduce multiple scheduler lifecycles and partially completed states.

## 5. Staged batch execution

The Brain still batches by the underlying decoder Binding, so parent operations in one batch may carry different head/tail Bindings:

```text
operation 0: head A -> decoder -> tail X
operation 1: carrier -> decoder
operation 2: head B -> decoder -> tail X
operation 3: carrier -> decoder -> tail Y
```

When a fused graph cannot be used, real execution proceeds as follows:

1. `_prepare_generating_inputs()` groups work by head Binding and executes zero or more head batches. Each group prefers a standalone head graph and falls back to eager. Operations without a head obtain hidden state from their carrier.
2. Inputs form one decoder batch in parent-operation order. A standalone decoder graph is preferred; LoRA, excessive context, and other unsupported cases use eager.
3. `_run_embedded_decode_tails()` groups work by tail Binding and executes zero or more tail batches. Greedy/no-logprobs groups prefer standalone tail graphs; other sampling is eager. Operations without a tail return a hidden-state carrier.

When the complete batch shares one head or tail, its batch tensor is reused without an extra `torch.stack`. Only heterogeneous batches are reorganized through per-operation views. LoRA/P-Tuning information still comes from the `SeqComplex` associated with each decoder Binding; cache, slot, and adapter-provider construction are unchanged.

## 6. Memory modeling

The external and internal lifecycle of a composite operation is:

| Form | External input | Internal temporaries | External result |
|---|---|---|---|
| decoder | carrier | decoder output | carrier |
| head + decoder | token | head output, decoder output | carrier |
| decoder + tail | carrier | decoder output | sampled token/logprobs |
| head + decoder + tail | token | head output, decoder output | sampled token/logprobs |

`MemoryBrain.do_generating()` must retain the peak occupancy of both embedded head output and decoder output. After `bind()`, head output is always released. Decoder output remains tracked by `Env` only when it leaves the parent operation as a device carrier; it is released when consumed by a tail. KV-slot, readonly-block, and adapter-block modeling remains centered on the decoder Binding.

## 7. Fused graphs and adjacency reconciliation

Standalone head, decoder, and tail graphs always remain available. `make_model_head()` and `make_model_tail()` may register an adjacent decoder Binding. After a related model load or adjacency registration, the Brain incrementally reconciles the decoder's fused variants:

| Adjacency | Fused variants |
|---|---|
| only `head -> D` | `head + D` |
| only `D -> tail` | `D + tail` |
| `head -> D -> tail` | only `head + D + tail` |

For example, registering the head can capture `head + D`; registering a tail for the same decoder later removes that variant and replaces it with the three-stage variant. The decoder-only graph is retained.

At runtime, fused replay occurs only when every selected operation has exactly the same optional head/tail structure and Bindings and the exact variant exists. A `head + D + tail` graph cannot serve a `head + D` request. Mixed head/tail structures, missing exact variants, LoRA, non-greedy sampling, or excessive context use the staged path, where eligible standalone stage graphs may still replay.

`GraphKey` includes decoder Binding, batch size, cache-engine runtime identity, and optional head/tail Bindings. Head/tail semantics do not use module `id()`. Requested bytes for each captured variant enter the MemoryBrain runtime pool independently and are removed when the variant is deleted.

## 8. Regression checks

After modifying this path, check at least:

- one Brain with head + decoder + tail at batch 1 and maximum concurrency;
- two Brains split 16/16, with head + decoder on the first and decoder + tail on the second;
- identical tokens through the same `InferNode`/shim/Brain path in offline and online modes;
- increasing `fused_replays` for exact-match graph batches;
- the expected fallback plus conditional standalone head/decoder/tail replay when an exact variant is absent;
- output-token hashes matching the standalone implementation;
- greedy, logprobs, LoRA, and P-Tuning eager requests;
- synchronized shim step/running state, KV-slot count, and `SequenceCore.output_token_ids`;
- `MemoryBrain` trial, revert, bind, and CPU/device carrier boundaries.
