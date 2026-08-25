# Experimental Container Implementations

These classes are imported by the parent `train.py` / `infer.py` scripts and are not standalone tests.

| File | Class | Behavior |
|---|---|---|
| `base.py` | `BaseContainer`, `Operate` | operation queue, execution modes, raw profiling, and waiting metrics |
| `fast.py` | `FastContainer` | blocking-scheduler baseline with minimal measurement/synchronization |
| `measure.py` | `MeasureContainer` | record CPU submission time per step and synchronize the stream to estimate CUDA time |
| `solve.py` | `SolveContainer` | collect real completion times and cumulative CPU/CUDA estimates to fit `alpha/beta` |
| `pressure.py` | `PressureReleaseContainer` | predict queued pressure from operation profiles and insert sleep plus CUDA-event synchronization |

The implementations contain experimental constants, including fixed `alpha=0.5327`, `beta=0.0191`, thresholds, and calibration overhead in `pressure.py`. Recalibrate through `measure -> solve -> pressure` after changing GPU, CPU, CUDA, or PyTorch versions.

Interfaces such as `data_device_to_cpu`, `data_cpu_to_device`, and `slow_operate` remain `NotImplementedError` in some implementations. Parent workloads may not currently exercise them, but verify call paths before extending the tests.
