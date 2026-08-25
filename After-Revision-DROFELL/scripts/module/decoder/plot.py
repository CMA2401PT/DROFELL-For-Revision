"""Plot decoder time/memory records captured by :mod:`test`.

The CUDA allocator snapshots contain an ordered allocation/free trace, but no
wall-clock timestamp for each allocator event.  Consequently the lifecycle
plot uses allocator event index on the x axis and reports measured wall-clock
stage time separately in the stage labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from operators.helpers.cuda_utils.memory import MemoryDeltaStatus


MIB = 2**20
STAGE_COLORS = {
    "forward": "#4477AA",
    "swap out": "#EE7733",
    "swap in": "#228833",
    "backward": "#CC3311",
}
BACKEND_COLORS = {
    "exllama_fuse": "#4477AA",
    "marlin": "#EE7733",
}


def _device_trace(snapshot: Mapping[str, Any] | None) -> list[Any] | None:
    if snapshot is None:
        return None
    traces = snapshot["device_traces"]
    if len(traces) != 1:
        raise ValueError(f"expected one visible CUDA device, got {len(traces)}")
    return traces[0]


def _mean_ms(samples_ns: Sequence[int]) -> float:
    return float(np.mean(samples_ns) / 1e6) if samples_ns else 0.0


@dataclass
class AnalysisRet:
    """Memory histories and timing statistics for one benchmark case."""

    forward: MemoryDeltaStatus
    backward: MemoryDeltaStatus
    swapout: MemoryDeltaStatus | None
    swapin: MemoryDeltaStatus | None
    forward_avg: float
    forward_std: float
    swap_avg: float
    swap_std: float
    backward_avg: float
    backward_std: float

    def _stages(self) -> list[tuple[str, MemoryDeltaStatus]]:
        stages: list[tuple[str, MemoryDeltaStatus]] = [("forward", self.forward)]
        if self.swapout is not None:
            stages.append(("swap out", self.swapout))
        if self.swapin is not None:
            stages.append(("swap in", self.swapin))
        stages.append(("backward", self.backward))
        return stages

    def stage_series(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Split cumulative snapshot histories into non-overlapping stages."""

        series: list[tuple[str, np.ndarray, np.ndarray]] = []
        previous: list[int] = []
        for stage, status in self._stages():
            history = status.history
            if history[: len(previous)] != previous:
                raise ValueError(f"{stage} snapshot does not extend the prior trace")
            start = len(previous)
            stage_values = history[start:]
            if start == 0:
                x = np.arange(0, len(stage_values) + 1)
                y = np.asarray([0, *stage_values], dtype=np.float64) / MIB
            else:
                # Include the boundary value in both adjacent colored lines so
                # the lifecycle stays visually continuous.
                x = np.arange(start, len(history) + 1)
                y = np.asarray([previous[-1], *stage_values], dtype=np.float64) / MIB
            series.append((stage, x, y))
            previous = history
        return series

    def plot(
        self,
        ax: Any | None = None,
        *,
        title: str | None = None,
        annotate: bool = True,
    ) -> Any:
        """Plot memory lifecycle and measured time for one case."""

        if ax is None:
            _, ax = plt.subplots(figsize=(10.5, 4.2))
        series = self.stage_series()
        timing = {
            "forward": self.forward_avg,
            "swap out": self.swap_avg / 2,
            "swap in": self.swap_avg / 2,
            "backward": self.backward_avg,
        }
        boundaries: list[int] = []
        for index, (stage, x, y) in enumerate(series):
            if index:
                boundaries.append(int(x[0]))
            label = stage
            if stage == "swap out":
                label = f"swap out/in ({self.swap_avg:.2f} ms total)"
            elif stage != "swap in":
                label = f"{stage} ({timing[stage]:.2f} ms)"
            else:
                label = "swap in"
            ax.plot(x, y, color=STAGE_COLORS[stage], linewidth=1.35, label=label)

        for boundary in boundaries:
            ax.axvline(boundary, color="#999999", linestyle=":", linewidth=0.8)

        if annotate:
            forward_peak_x = self.forward.peak_step + 1
            forward_peak = self.forward.peak_memory / MIB
            residual_x = len(self.forward.history)
            residual = self.forward.preserved_new_alloc / MIB
            backward_peak_x = self.backward.peak_step + 1
            backward_peak = self.backward.peak_memory / MIB
            annotations = (
                (forward_peak_x, forward_peak, f"Fwd peak {forward_peak:.2f} MiB"),
                (residual_x, residual, f"Fwd residual {residual:.2f} MiB"),
                (backward_peak_x, backward_peak, f"Bwd peak {backward_peak:.2f} MiB"),
            )
            for x_value, y_value, text in annotations:
                ax.scatter([x_value], [y_value], s=13, color="#222222", zorder=4)
                ax.annotate(
                    text,
                    (x_value, y_value),
                    xytext=(5, 7),
                    textcoords="offset points",
                    fontsize=8,
                    color="#222222",
                )

        ax.set_xlim(0, max(len(self.backward.history), 1))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("CUDA allocator event (allocation/free order; not wall-clock time)")
        ax.set_ylabel("Incremental requested memory (MiB)")
        if title:
            ax.set_title(title)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
        return ax


def analysis_forward_backward_memory(
    forward_trace: Sequence[Any],
    backward_trace: Sequence[Any],
    swap_out_trace: Sequence[Any] | None,
    swap_in_trace: Sequence[Any] | None,
    forward_time: Sequence[int],
    swap_time: Sequence[int],
    backward_time: Sequence[int],
) -> AnalysisRet:
    """Build the historical plotting result object from raw record fields."""

    forward = MemoryDeltaStatus(list(forward_trace))
    if forward.free_before != 0:
        raise ValueError(f"forward trace frees {forward.free_before} pre-history bytes")
    swapout = MemoryDeltaStatus(list(swap_out_trace)) if swap_out_trace else None
    swapin = MemoryDeltaStatus(list(swap_in_trace)) if swap_in_trace else None
    backward = MemoryDeltaStatus(list(backward_trace))

    def std_ms(values: Sequence[int]) -> float:
        return float(np.std(values) / 1e6) if values else 0.0

    return AnalysisRet(
        forward=forward,
        swapout=swapout,
        swapin=swapin,
        backward=backward,
        forward_avg=_mean_ms(forward_time),
        forward_std=std_ms(forward_time),
        swap_avg=_mean_ms(swap_time),
        swap_std=std_ms(swap_time),
        backward_avg=_mean_ms(backward_time),
        backward_std=std_ms(backward_time),
    )


def analysis_from_record(record: Mapping[str, Any]) -> AnalysisRet:
    """Build plotting data from the current or historical record schema."""

    forward_trace = _device_trace(record["forward_memory"])
    backward_trace = _device_trace(record["backward_memory"])
    assert forward_trace is not None and backward_trace is not None
    return analysis_forward_backward_memory(
        forward_trace=forward_trace,
        swap_out_trace=_device_trace(record.get("swap_out_memory")),
        swap_in_trace=_device_trace(record.get("swap_in_memory")),
        backward_trace=backward_trace,
        forward_time=record["forward_time"],
        swap_time=record.get("swap_time", []),
        backward_time=record["backward_time"],
    )


def plot_record(
    record: Mapping[str, Any],
    output: Path,
    *,
    title: str | None = None,
    dpi: int = 240,
) -> None:
    """Write one case's time-annotated memory lifecycle plot."""

    result = analysis_from_record(record)
    figure, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    result.plot(ax, title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _comparison_plot(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    metrics: Sequence[tuple[str, str]],
    title: str,
    paper: bool,
    log_metrics: frozenset[str] = frozenset(),
    dpi: int = 240,
) -> None:
    strategies = list(dict.fromkeys(str(row["paper_name"]) for row in rows))
    backends = list(dict.fromkeys(str(row["backend"]) for row in rows))
    x = np.arange(len(strategies), dtype=np.float64)
    width = 0.72 / max(len(backends), 1)
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.1 * len(metrics), 3.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    by_key = {
        (str(row["backend"]), str(row["paper_name"])): row for row in rows
    }
    for metric_index, ((metric, label), ax) in enumerate(zip(metrics, axes)):
        for backend_index, backend in enumerate(backends):
            selected = [by_key[(backend, strategy)] for strategy in strategies]
            values = np.asarray([float(row[metric]) for row in selected])
            errors = None
            error_key = metric.replace("_ms", "_stdev_ms")
            if error_key != metric and error_key in selected[0]:
                errors = np.asarray([float(row[error_key]) for row in selected])
            offset = (backend_index - (len(backends) - 1) / 2) * width
            backend_label = "ExLlama" if backend == "exllama_fuse" else backend.title()
            ax.bar(
                x + offset,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                label=backend_label,
                color=BACKEND_COLORS.get(backend),
                alpha=0.9,
            )
        if paper:
            paper_key = f"paper_{metric}"
            paper_values = [
                by_key[(backends[0], strategy)].get(paper_key)
                for strategy in strategies
            ]
            if all(value is not None for value in paper_values):
                ax.plot(
                    x,
                    paper_values,
                    color="#222222",
                    marker="x",
                    linestyle="--",
                    linewidth=1,
                    label="Paper (ExLlama)",
                )
        ax.set_title(label)
        ax.set_xticks(x, strategies)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        if metric in log_metrics:
            ax.set_yscale("log")
            ax.set_ylabel("log scale")
        if metric_index == 0:
            ax.legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_run_comparisons(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, *, dpi: int = 240
) -> None:
    """Write backend/strategy timing and memory comparison figures."""

    _comparison_plot(
        rows,
        output_dir / "timing-comparison.png",
        metrics=(
            ("forward_ms", "Forward time (ms)"),
            ("swap_ms", "Swap out + in (ms)"),
            ("backward_ms", "Backward time (ms)"),
        ),
        title="Decoder stage time by backend and checkpoint strategy",
        paper=True,
        log_metrics=frozenset({"swap_ms"}),
        dpi=dpi,
    )
    _comparison_plot(
        rows,
        output_dir / "memory-comparison.png",
        metrics=(
            ("forward_peak_mib", "Forward peak (MiB)"),
            ("forward_residual_mib", "Forward residual (MiB)"),
            ("backward_peak_mib", "Backward peak (MiB)"),
        ),
        title="Incremental requested memory by backend and checkpoint strategy",
        paper=True,
        log_metrics=frozenset({"forward_residual_mib"}),
        dpi=dpi,
    )
