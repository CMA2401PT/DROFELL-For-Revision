import pandas as pd
from typing import Any, Dict, List, Tuple, Set
from typing import Set
from typing import Any, Dict, List
from numpy import argmax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TraceRecord:
    def __init__(self, traces: List[Any]) -> None:
        self.free_before, self.current, self.history =\
            self.analyzed_trace(traces)
        self.data_block = self.gen_block_history(traces)
        self.peak_memory = max(self.history)
        self.peak_step = int(argmax(self.history))

    @staticmethod
    def analyzed_trace(traces: List[Any]) -> Tuple[int, int, List[int]]:
        current: Dict[int, int] = {}
        free_before: Dict[int, int] = {}
        history: List[int] = []
        for trace in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action == 'alloc':
                current[addr] = size
                history.append(sum(current.values()))
            if action == "free_requested":
                continue
            if action == "free_completed":
                if not addr in current:
                    free_before[addr] = size
                else:
                    del current[addr]
                    history.append(sum(current.values()))
            if action == "snapshot":
                continue
            assert "frames" in trace, trace
            assert len(trace["frames"])
        return sum(free_before.values()), sum(current.values()), history

    @staticmethod
    def gen_block_history(traces: List[Any]):
        occoured: Set[int] = set()
        steps = 0
        for trace in traces:
            addr = trace["addr"]
            action = trace["action"]
            if action == 'alloc':
                occoured.add(addr)
                steps += 1
            elif action == "free_completed":
                steps += 1

        addrs = [a for a in occoured]
        addrs = sorted(addrs)
        addr_mapping = {a: i for i, a in enumerate(addrs)}
        data_block = np.zeros((len(addrs), steps), dtype=np.int64)

        current: Dict[int, int] = {}
        step_i = 0
        for (trace) in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action not in ['alloc', 'free_completed']:
                continue

            if action == 'alloc':
                current[addr] = size
                for addr, size in current.items():
                    data_block[addr_mapping[addr], step_i] = size
            elif action == "free_completed":
                if addr in current:
                    del current[addr]

            for addr, size in current.items():
                data_block[addr_mapping[addr], step_i] = size
            step_i += 1
        return data_block

    def plot(self):

        _, steps = self.data_block.shape
        # set seaborn style
        sns.set_theme(context='paper', style='whitegrid', font="Times New Roman")
        plt.rcParams["patch.force_edgecolor"] = False
        x = range(steps)
        y = self.data_block/(2**10)

        # Plot
        plt.stackplot(x, y)
        plt.xlim(-10, steps+10)
        plt.xlabel("Operate")
        plt.ylabel("Memory")


def extract_speed_multi(record: List[Tuple[int, int]]) -> List[float]:
    token_time: List[Tuple[float, int]] = []
    i_seq = record[0][0]
    st: int = record[0][1]
    tks = 1
    for seq_i, t in record[1:]:
        if seq_i == i_seq:
            token_time.append(((t-st)/(10**9), tks))
            st = t
            tks = 1
        else:
            tks += 1
    speed: List[float] = []
    for dt, f in token_time:
        speed.append(f/dt)
    return speed


def plot_speed(records: Dict[Tuple[str, str], List[Tuple[int, int]]]):
    pd_records: List[Dict[str, Any]] = []
    for (rec_name, rec_source), record in records.items():
        speed = extract_speed_multi(record)
        for i, s in enumerate(speed):
            pd_records.append({"Seq Length": (i//8*8), "Parallel": rec_name, "Tokens Per Second": s, "Method": rec_source})
    sns.set_style("whitegrid")
    sns.set_theme(font="Times New Roman")
    # plt.figure(figsize=(8,5))
    data = pd.DataFrame.from_records(pd_records)
    sns.lineplot(data, x="Seq Length", y="Tokens Per Second", hue="Parallel", style="Method")

# %%


def plot_speed_ratio(records: Dict[str, Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]):
    pd_records: List[Dict[str, Any]] = []

    def extract_speed(record: List[Tuple[int, int]]):
        token_time: List[Tuple[float, int]] = []
        st = record[0][1]
        curr: Set[int] = set([record[0][0]])
        for seq_i, t in record[1:]:
            if seq_i in curr:
                token_time.append(((t-st)/(10**9), len(curr)))
                curr = set([seq_i])
                st = t
            else:
                curr.add(seq_i)
        speed: List[float] = []
        for dt, f in token_time:
            speed.append(f/dt)
        return speed
    for rec_name, (orig_record, new_record) in records.items():
        orig_speed = extract_speed(orig_record)
        new_speed = extract_speed(new_record)
        speed_ratio = [n/orig_speed[i] for i, n in enumerate(new_speed)]
        for i, s in enumerate(speed_ratio):
            pd_records.append({"Seq Len": (i//32*32), "Parallel": rec_name, "Ratio": s})
    sns.set_style("darkgrid")
    sns.set_theme(font="Times New Roman")
    # plt.figure(figsize=(8,5))
    data = pd.DataFrame.from_records(pd_records)
    sns.lineplot(data, x="Seq Len", y="Ratio", hue="Parallel")


def compute_avg_speed(record: List[Tuple[int, int]], ranges: List[int]):
    result: List[float] = []
    for r in ranges:
        if r == -1:
            result.append(len(record)/((record[-1][1]-record[0][1])/10**9))
        else:
            c: Dict[int, int] = {}
            for (i, end_t) in record:
                if not i in c:
                    c[i] = 1
                else:
                    c[i] += 1
                if min(c.values()) == r:
                    result.append(sum(c.values())/((end_t-record[0][1])/(10**9)))
                    break
    return result
