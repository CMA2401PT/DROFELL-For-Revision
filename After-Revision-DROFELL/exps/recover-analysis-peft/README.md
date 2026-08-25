# PEFT Recovery Experiments

This directory contains:

- [`basic/`](basic/): training recovery without a redundant path;
- [`multi/`](multi/): multi-path training recovery inside a Backbone;
- [`prob/`](prob/): single-machine training determinism and backend-difference isolation.

## Common semantics

Normal and Recover should fix data order, seed, adapter, optimizer, scheduler, precision, and submission cadence. Align comparisons by semantic minibatch/version, not wall-clock row number.

Equal scalar loss does not establish equal state trajectories. Record together:

- adapter/optimizer/scheduler state fingerprints;
- forward/backward boundary-tensor fingerprints;
- version, minibatch, and checkpoints;
- recovery events and valid output;
- loss, parameter differences, and final task metrics.

Generated results, logs, fingerprints, tables, and figures are not version-controlled.
