#!/usr/bin/env python3
"""Aggregate a completed Binding/async/merge experiment run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


SEQ_COUNTS = (1, 8, 16)
DEPLOYMENTS = ("persistent-agent", "combined-request-agent")
ONLINE_CASES = {
    "async-py-sync-single": {
        "async_backend": "py", "busy_wait": False, "container": "nonasync",
        "binding_size": 16, "merge_limit": 8, "axes": ("async",),
    },
    "default-go-blocking-dual": {
        "async_backend": "golang", "busy_wait": False, "container": "shared",
        "binding_size": 16, "merge_limit": 8,
        "axes": ("async", "binding", "merge"),
    },
    "async-go-poll-dual": {
        "async_backend": "golang", "busy_wait": True, "container": "shared",
        "binding_size": 16, "merge_limit": 8, "axes": ("async",),
    },
    "binding-32": {
        "async_backend": "golang", "busy_wait": False, "container": "shared",
        "binding_size": 32, "merge_limit": 8, "axes": ("binding",),
    },
    "binding-8": {
        "async_backend": "golang", "busy_wait": False, "container": "shared",
        "binding_size": 8, "merge_limit": 8, "axes": ("binding",),
    },
    "merge-4": {
        "async_backend": "golang", "busy_wait": False, "container": "shared",
        "binding_size": 16, "merge_limit": 4, "axes": ("merge",),
    },
    "merge-16": {
        "async_backend": "golang", "busy_wait": False, "container": "shared",
        "binding_size": 16, "merge_limit": 16, "axes": ("merge",),
    },
}


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise AnalysisError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = report.get("cases")
    if isinstance(cases, list):
        return cases
    if isinstance(cases, dict):
        return list(cases.values())
    raise AnalysisError("result cases must be a list or object")


def round_ms(case: dict[str, Any]) -> float:
    summary = case["summary"]
    if "decode_round_ms_median" in summary:
        return float(summary["decode_round_ms_median"])
    values = []
    for run in case["runs"]:
        sample = run.get("decode", run)
        values.append(float(sample["avg_ms_per_logical_round"]))
    return statistics.median(values)


def performance_rows(
    report: dict[str, Any],
    *,
    implementation: str,
    category: str,
    online_case: str = "",
    deployment: str = "",
    binding_size: int | str = "",
    merge_limit: int | str = "",
    async_backend: str = "",
    busy_wait: bool | str = "",
    container: str = "",
) -> list[dict[str, Any]]:
    rows = []
    observed = set()
    for case in normalized_cases(report):
        seqs = int(case["num_seqs"])
        observed.add(seqs)
        summary = case["summary"]
        rows.append({
            "category": category,
            "implementation": implementation,
            "online_case": online_case,
            "deployment": deployment,
            "async_backend": async_backend,
            "busy_wait": busy_wait,
            "container": container,
            "binding_size": binding_size,
            "merge_limit": merge_limit,
            "num_seqs": seqs,
            "decode_tok_s_median": float(summary["decode_tok_s_median"]),
            "decode_tok_s_min": float(summary["decode_tok_s_min"]),
            "decode_tok_s_max": float(summary["decode_tok_s_max"]),
            "decode_round_ms_median": round_ms(case),
            "repeat_count": len(case["runs"]),
        })
    if observed != set(SEQ_COUNTS):
        raise AnalysisError(
            f"{implementation} has sequence counts {sorted(observed)}, "
            f"expected {list(SEQ_COUNTS)}"
        )
    return rows


def load_performance(root: Path) -> list[dict[str, Any]]:
    rows = []
    rows.extend(performance_rows(
        read_json(root / "baseline" / "base" / "result.json"),
        implementation="base", category="system",
    ))
    for binding_size in (16, 32):
        case_id = f"offline-binding-{binding_size}"
        rows.extend(performance_rows(
            read_json(root / "baseline" / case_id / "result.json"),
            implementation=case_id, category="system",
            binding_size=binding_size, merge_limit=8,
            async_backend="golang", busy_wait=False, container="nonasync",
        ))
    rows.extend(performance_rows(
        read_json(root / "baseline" / "vllm" / "result.json"),
        implementation="vllm", category="system",
    ))
    for case_id, config in ONLINE_CASES.items():
        for deployment in DEPLOYMENTS:
            report = read_json(
                root / "online" / case_id / deployment / "result.json"
            )
            rows.extend(performance_rows(
                report,
                implementation=f"online-{deployment}",
                category="online",
                online_case=case_id,
                deployment=deployment,
                binding_size=config["binding_size"],
                merge_limit=config["merge_limit"],
                async_backend=config["async_backend"],
                busy_wait=config["busy_wait"],
                container=config["container"],
            ))
    return rows


def gpu_power(path: Path, *, required: bool = False) -> dict[str, Any]:
    values = []
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace", newline="") as file:
            for row in csv.reader(file):
                if len(row) < 7:
                    continue
                try:
                    values.append(float(row[6].strip()))
                except ValueError:
                    continue
    if required and not values:
        raise AnalysisError(f"no numeric GPU power samples: {path}")
    steady_values = values[-min(20, len(values)):] if values else []
    return {
        "gpu_power_sample_count": len(values),
        "gpu_power_mean_w": statistics.fmean(values) if values else None,
        "gpu_power_median_w": statistics.median(values) if values else None,
        "gpu_power_max_w": max(values) if values else None,
        "gpu_power_steady_tail_sample_count": len(steady_values),
        "gpu_power_steady_tail_mean_w": (
            statistics.fmean(steady_values) if steady_values else None
        ),
        "gpu_power_steady_tail_median_w": (
            statistics.median(steady_values) if steady_values else None
        ),
    }


def resource_row(
    root: Path,
    case_id: str,
    deployment: str,
    scope: str,
) -> dict[str, Any]:
    case_root = root / "online" / case_id / deployment
    prefix = "idle_" if scope == "loaded-idle" else ""
    metrics = read_json(case_root / f"{prefix}process_metrics.json")
    wall = float(metrics["wall_seconds"])
    voluntary = int(metrics["voluntary_context_switches"])
    involuntary = int(metrics["involuntary_context_switches"])
    power = gpu_power(
        case_root / f"{prefix}gpu_monitor.csv",
        required=True,
    )
    config = ONLINE_CASES[case_id]
    return {
        "scope": scope,
        "online_case": case_id,
        "deployment": deployment,
        "async_backend": config["async_backend"],
        "busy_wait": config["busy_wait"],
        "container": config["container"],
        "wall_seconds": wall,
        "average_cpu_cores": metrics["average_cpu_cores"],
        "voluntary_context_switches_per_s": voluntary / wall,
        "involuntary_context_switches_per_s": involuntary / wall,
        "scheduler_context_switches_per_s": (voluntary + involuntary) / wall,
        **power,
    }


def load_resources(root: Path) -> list[dict[str, Any]]:
    rows = []
    async_cases = [
        case_id for case_id, config in ONLINE_CASES.items()
        if "async" in config["axes"]
    ]
    for case_id in async_cases:
        for deployment in DEPLOYMENTS:
            rows.append(resource_row(root, case_id, deployment, "loaded-idle"))
            rows.append(resource_row(root, case_id, deployment, "decode-suite"))
    return rows


def axis_rows(performance: Iterable[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    rows = []
    for row in performance:
        case_id = str(row["online_case"])
        if not case_id or axis not in ONLINE_CASES[case_id]["axes"]:
            continue
        rows.append(dict(row))
    if len(rows) != 3 * len(DEPLOYMENTS) * len(SEQ_COUNTS):
        raise AnalysisError(f"incomplete {axis} sensitivity table: {len(rows)} rows")
    if axis == "binding":
        rows.sort(key=lambda row: (
            str(row["deployment"]), int(row["binding_size"]), int(row["num_seqs"])
        ))
    elif axis == "merge":
        rows.sort(key=lambda row: (
            str(row["deployment"]), int(row["merge_limit"]), int(row["num_seqs"])
        ))
    else:
        order = {case_id: index for index, case_id in enumerate(
            ("async-py-sync-single", "default-go-blocking-dual", "async-go-poll-dual")
        )}
        rows.sort(key=lambda row: (
            str(row["deployment"]), order[str(row["online_case"])], int(row["num_seqs"])
        ))
    return rows


def system_rows(performance: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in performance:
        if row["category"] == "system":
            selected.append(dict(row))
        elif row["online_case"] == "default-go-blocking-dual":
            selected.append(dict(row))
    expected = 6 * len(SEQ_COUNTS)
    if len(selected) != expected:
        raise AnalysisError(f"incomplete system comparison: {len(selected)} rows")
    return selected


def wide_performance_rows(
    rows: Iterable[dict[str, Any]],
    identity_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    observed: dict[tuple[Any, ...], set[int]] = {}
    for row in rows:
        identity = tuple(row[key] for key in identity_keys)
        target = grouped.setdefault(
            identity,
            {key: row[key] for key in identity_keys},
        )
        seqs = int(row["num_seqs"])
        if seqs in observed.setdefault(identity, set()):
            raise AnalysisError(
                f"duplicate sequence count {seqs} for {identity}"
            )
        observed[identity].add(seqs)
        target[f"seq_{seqs}_tok_s_median"] = row["decode_tok_s_median"]
        target[f"seq_{seqs}_round_ms_median"] = row[
            "decode_round_ms_median"
        ]
    expected = set(SEQ_COUNTS)
    for identity, actual in observed.items():
        if actual != expected:
            raise AnalysisError(
                f"incomplete sequence counts for {identity}: "
                f"observed={sorted(actual)}, expected={sorted(expected)}"
            )
    return list(grouped.values())


PERFORMANCE_WIDE_COLUMNS = [
    item
    for seqs in SEQ_COUNTS
    for item in (
        (f"seq_{seqs}_tok_s_median", f"Seq={seqs} tok/s"),
        (f"seq_{seqs}_round_ms_median", f"Seq={seqs} ms/round"),
    )
]


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output: Path,
    system_wide: list[dict[str, Any]],
    async_wide: list[dict[str, Any]],
    binding_wide: list[dict[str, Any]],
    merge_wide: list[dict[str, Any]],
    resources: list[dict[str, Any]],
) -> None:
    lines = [
        "# Binding / async / merge 实验报告", "",
        "全部结果使用 CUDA Graph；吞吐为所有并发序列的合计 output tokens/s。", "",
        "## 系统性能", "",
    ]
    lines.extend(markdown_table(
        system_wide,
        [("implementation", "实现"), *PERFORMANCE_WIDE_COLUMNS],
    ))
    for title, rows, columns in (
        ("3.10 Async 路径", async_wide, [
            ("online_case", "设置"), ("deployment", "部署"),
            *PERFORMANCE_WIDE_COLUMNS,
        ]),
        ("2.16 Binding 粒度", binding_wide, [
            ("binding_size", "Binding size"), ("deployment", "部署"),
            *PERFORMANCE_WIDE_COLUMNS,
        ]),
        ("2.18 Merge limit", merge_wide, [
            ("merge_limit", "Merge limit"), ("deployment", "部署"),
            *PERFORMANCE_WIDE_COLUMNS,
        ]),
        ("3.10 CPU 与功耗代理", resources, [
            ("scope", "范围"), ("online_case", "设置"),
            ("deployment", "部署"), ("average_cpu_cores", "CPU cores"),
            ("scheduler_context_switches_per_s", "context switches/s"),
            ("gpu_power_steady_tail_mean_w", "GPU power steady-tail W"),
        ]),
    ):
        lines.extend(("", f"## {title}", ""))
        lines.extend(markdown_table(rows, columns))
    lines.extend((
        "", "## 解释边界", "",
        "- `py+False+NonAsyncContainer` 是 CUDA 工作绕过 blocking queue 的单队列执行路径；事件循环实现本身仍保留双队列代码。",
        "- 不存在“单队列 + polling”实验格：同步 CUDA 路径只能调用同步等待。",
        "- 单 Dealer Binding 扫描只支持本地吞吐和延迟敏感性结论，不能替代迁移、恢复或跨节点通信实验。",
        "- Linux context switch 是 OS 调度代理，不能写成事件循环唤醒次数。",
        "- loaded-idle 的全窗口 GPU power 包含 Graph capture 后的降频冷却；稳态列取最后 20 个约 0.5 秒样本。该机器未暴露 CPU package/整机功耗计数器。",
    ))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = args.output.resolve() if args.output else root / "analysis"
    performance = load_performance(root)
    resources = load_resources(root)
    async_rows = axis_rows(performance, "async")
    binding_rows = axis_rows(performance, "binding")
    merge_rows = axis_rows(performance, "merge")
    system = system_rows(performance)
    system_wide = wide_performance_rows(system, ("implementation",))
    async_wide = wide_performance_rows(
        async_rows, ("online_case", "deployment")
    )
    binding_wide = wide_performance_rows(
        binding_rows, ("binding_size", "deployment")
    )
    merge_wide = wide_performance_rows(
        merge_rows, ("merge_limit", "deployment")
    )
    write_csv(output / "performance_all.csv", performance)
    write_csv(output / "system_performance.csv", system)
    write_csv(output / "async_sensitivity.csv", async_rows)
    write_csv(output / "binding_sensitivity.csv", binding_rows)
    write_csv(output / "merge_sensitivity.csv", merge_rows)
    write_csv(output / "async_resource_overhead.csv", resources)
    write_csv(output / "system_performance_wide.csv", system_wide)
    write_csv(output / "async_sensitivity_wide.csv", async_wide)
    write_csv(output / "binding_sensitivity_wide.csv", binding_wide)
    write_csv(output / "merge_sensitivity_wide.csv", merge_wide)
    write_json(output / "summary.json", {
        "schema_version": 1,
        "result_root": str(root),
        "row_counts": {
            "performance_all": len(performance),
            "system": len(system),
            "async": len(async_rows),
            "binding": len(binding_rows),
            "merge": len(merge_rows),
            "resources": len(resources),
            "system_wide": len(system_wide),
            "async_wide": len(async_wide),
            "binding_wide": len(binding_wide),
            "merge_wide": len(merge_wide),
        },
        "interpretation_limits": [
            "The py/nonasync cell bypasses the blocking queue but does not replace the event loop with a separately implemented single-queue loop.",
            "The one-Dealer Binding scan does not measure migration, recovery, or network communication cost.",
            "Linux context switches are not event-loop wakeup counts.",
        ],
    })
    write_report(
        output,
        system_wide,
        async_wide,
        binding_wide,
        merge_wide,
        resources,
    )
    print(json.dumps({"output": str(output), "rows": len(performance)}, indent=2))


if __name__ == "__main__":
    main()
