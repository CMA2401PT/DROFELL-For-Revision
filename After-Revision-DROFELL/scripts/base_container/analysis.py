import json
from collections import defaultdict
import os
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_data(json_file: str) -> List[Tuple[str, int, int]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data  # Should be List[Tuple[str, int, int]]


def compute_exp_base_waiting_metrics(seq: List[Tuple[float, float]]):
    """
    Computes average expected wait (over whole busy periods) and worst-case wait.
    """
    total_time = sum(w + i for w, i in seq)

    if total_time == 0:
        return 0, 0, 0  # avoid division by zero

    # Average: weighted average over busy periods
    weighted_sum = sum((w * (w / 2)) for w, _ in seq)  # average wait in each busy segment
    average_wait = weighted_sum / total_time

    # Worst-case: maximum duration of any busy period
    worst_wait = np.percentile([w for w, _ in seq], 99.85)
    busy_percentage = (sum(w for w, _ in seq)/total_time)
    return average_wait, worst_wait, busy_percentage


def compute_statsbase_waiting_metrics(event_stats: List[Dict[str, float]]):
    """
    event_stats: list of dicts, each with keys:
        'count': int, number of times the event type occurs (n_k)
        'mean_w': float, mean duration (mu_wk)
        'std_w': float, std dev of duration (sigma_wk)
        'mean_i': float, mean idle (mu_ik)
        'std_i': float, std dev of idle (sigma_ik)
    Returns:
        average_wait: float, expected waiting time when sampling t uniformly
        worst_case_wait: float, max possible waiting time (approx mu + 3σ)
    """
    total_time = sum(e['count'] * (e['mean_w'] + e['mean_i']) for e in event_stats)

    weighted_w2_sum = sum(e['count'] * (e['mean_w'] ** 2) for e in event_stats)

    average_wait = weighted_w2_sum / (2 * total_time)

    worst_case_wait = max(e['mean_w'] + 3 * e['std_w'] for e in event_stats)

    return average_wait, worst_case_wait


def compute_stats(data: List[Tuple[str, int, int]]):
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"cpu": [], "cuda": [], "total": []})
    seq: List[Tuple[float, float]] = []
    first = set()
    for type_name, cpu_time, cuda_time in data:
        if not type_name in first:
            first.add(type_name)
            continue
        grouped[type_name]["cpu"].append(cpu_time/(10**6))
        grouped[type_name]["cuda"].append(cuda_time/(10**6))
        grouped[type_name]["total"].append((cpu_time+cuda_time)/(10**6))
        seq.append((cpu_time/(10**6), cuda_time/(10**6)))

    average_wait, worst_wait, busy_percentage = compute_exp_base_waiting_metrics(seq)
    print("Exp based")
    print(f"\tPer Blocking Time: {sum(w for w, _ in seq)/len(seq):.4f}ms")
    print(f"\tAverage Actual wait: {average_wait:.4f}ms")
    print(f"\tWorst-case Actual wait: {worst_wait:.4f}ms")
    print(f"\tBusy Percentage: {busy_percentage:.2%}")
    result: List[Dict[str, Any]] = []
    for type_name, ops in grouped.items():
        cpu_list = ops["cpu"]
        cuda_list = ops["cuda"]
        total_list = ops["total"]
        cpu_mean = mean(cpu_list)
        cpu_stdev = stdev(cpu_list) if len(cpu_list) > 1 else 0.0

        cuda_mean = mean(cuda_list)
        cuda_stdev = stdev(cuda_list) if len(cuda_list) > 1 else 0.0

        total_mean = mean(total_list)
        total_stdev = stdev(total_list) if len(total_list) > 1 else 0.0

        # if cpu_mean < 1 and cuda_mean < 1:
        #     continue
        result.append({
            "type": type_name,
            "cpu_mean": cpu_mean,
            "cpu_total": sum(cpu_list),
            "cpu_stdev": cpu_stdev,
            "cuda_mean": cuda_mean,
            "cuda_stdev": cuda_stdev,
            "cuda_total": sum(cuda_list),
            "total_mean": total_mean,
            "total_stdev": total_stdev,
            "count": len(cpu_list),
        })
    return result, [w for w, _ in seq]


def plot_stats(stats_list: List[Dict[str, Any]], waits: List[float], out_file: str, sort_by: str = 'cpu_mean'):
    # 按排序字段排序
    sorted_stats = sorted(stats_list, key=lambda x: x[sort_by], reverse=False)
    types = [item['type'] for item in sorted_stats]

    cpu_means = [item['cpu_mean'] for item in sorted_stats]
    cpu_stdevs = [item['cpu_stdev'] for item in sorted_stats]

    cuda_means = [item['cuda_mean'] for item in sorted_stats]
    cuda_stdevs = [item['cuda_stdev'] for item in sorted_stats]

    total_mean = [item['total_mean'] for item in sorted_stats]
    total_stdevs = [item['total_stdev'] for item in sorted_stats]

    average_wait, worst_case_wait = compute_statsbase_waiting_metrics([
        {
            "count": e["count"],
            "mean_w": e["cpu_mean"],
            "std_w": e["cpu_stdev"],
            "mean_i": e["cuda_mean"],
            "std_i": e["cuda_stdev"],
        } for e in stats_list
    ])
    print("Static Based")
    print(f"\tAverage expected wait: {average_wait:.4f}ms")
    print(f"\tWorst-case wait: {worst_case_wait:.4f}ms")

    for t_name, cpu_mean, cuda_mean in zip(types, cpu_means, cuda_means):
        print(f"\t{cpu_mean:.2f}+{cuda_mean:.2f}, {t_name}")

    s = "pressure={"
    for e in stats_list:
        name = (e['type'])
        s += f"'{name}':({e['cpu_mean']},{e['cpu_stdev']},{e['cuda_mean']},{e['cuda_stdev']}),\n"
    s += "}"
    with open(f"{out_file}_S.py", "w") as f:
        f.write(s)
    y_pos = range(len(types))

    fig, ax = plt.subplots(figsize=(20, len(types) * 1))

    ax.errorbar(cpu_means, y_pos, xerr=cpu_stdevs, fmt='o', label='cpu', color='blue', capsize=3)
    ax.errorbar(cuda_means, y_pos, xerr=cuda_stdevs, fmt='o', label='cuda', color='green', capsize=3)
    # ax.errorbar(total_mean, y_pos, xerr=total_stdevs, fmt='o', label='total', color='red', capsize=3)

    # ax.set_ylim(min(y_pos)-1, max(y_pos)+1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Average Time per Type with Standard Deviation')
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    # ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_file+".png", dpi=330)

    # plt.figure()
    # x = np.linspace(min(waits), max(waits), 100)
    # plt.hist(waits, bins=30, density=True, alpha=0.6, color='skyblue', label='Residuals Histogram')
    # plt.xlabel("Wait time")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid()
    # plt.savefig(out_file+"_bin.png")


if __name__ == "__main__":
    for f in os.listdir(os.path.join(os.path.dirname(__file__), "results")):
        if not f.endswith(".json"):
            continue
        if f == "model_record.json":
            continue
        profile = f.removesuffix(".json")
        print(profile)
        json_file = os.path.join(os.path.dirname(__file__), "results", f"{profile}.json")  # 你的 JSON 文件路径
        data = load_data(json_file)
        stats, waits = compute_stats(data)
        plot_stats(
            stats, waits,
            out_file=os.path.join(os.path.dirname(__file__), "results", f"{profile}"),
            sort_by='cpu_mean')
