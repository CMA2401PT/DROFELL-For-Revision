# Inference Algorithm Interfaces and Data Structures

This directory has no executable entry point.

| Module | Primary contents |
|---|---|
| `basic_types.py` | aliases for token, sequence, virtual device/block, and related types |
| `sequence.py` | `Sequence`, `SequenceGroup`, state, and token data |
| `sampling_params.py`, `sample.py` | sampling parameters, metadata, sampler abstractions, and output |
| `block.py` | logical blocks and virtual physical blocks |
| `block_space.py`, `block_space_io.py` | block-holder/I/O abstractions and allocation state |
| `scheduler.py`, `policy.py` | scheduler and ordering-policy interfaces |
| `meta.py` | prefill/generating execution metadata and memory operations |
| `ptuning.py` | P-Tuning block handler |
| `infer_request_outputs.py` | completion/request output objects |

After modifying interfaces, regress the two parent sample requests, 1/2/4/8-concurrency decode, and single/multi prefill; these paths cover different interface subsets.
