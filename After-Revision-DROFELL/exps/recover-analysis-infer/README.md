# Inference Recovery Experiments

This directory contains two complementary protocols:

- [`basic/`](basic/): detection, logical handoff, physical reconstruction, and output recovery after a single-path failure;
- [`multi/`](multi/): interaction between surviving and failed paths under multi-path dispatch.

The protocols are not interchangeable. Basic decomposes recovery stages; Multi tests whether a redundant path continues providing boundary output during physical recovery.

## Common requirements

- Fix model, sampling, input tokens, topology, and seed.
- Choose the failure target from the actual source of the latest accepted output, not a fixed node name.
- Record failure request, disconnect, replacement-graph installation, logical handoff, physical recovery, and first valid output.
- Compare token sequences and confirmed prefixes between Normal and Recover.
- Record eager/graph, checkpoint, and replay modes explicitly.
- Use an independent run ID for every run; do not version-control generated data.

## Output boundary

Analyzers may generate event tables, timelines, machine-check summaries, and figures, but the documentation retains no existing values. Do not aggregate output across different profiles, path latencies, checkpoints, or failure counts.
