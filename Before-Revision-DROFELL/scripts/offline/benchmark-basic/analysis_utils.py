from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing import Dict, List

import pandas as pd
import seaborn as sns


def compute_avg_speed(record: List[Tuple[int, int]], ranges: Optional[List[int]] = None) -> List[float]:
    result: List[float] = []
    if ranges is None:
        ranges = [100, 1000, 2000, -1]
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


def extract_speed_base(record: List[Tuple[int, int]]):
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


def extract_speed_single(record: List[int]):
    return extract_speed_base([(0, r) for r in record])


def extract_speed(record: Union[List[int], List[Tuple[int, int]]]):
    if isinstance(record[0], int):
        return extract_speed_single(record)  # type: ignore
    else:
        return extract_speed_base(record)  # type: ignore


def smooth_speed(speed: List[float], window: int) -> List[float]:
    s_out: List[float] = []
    for i in range(0, len(speed)-window, window):
        s_out.append(sum(speed[i:i+window])/window)
    return s_out


class Analyzer:
    def __init__(self, limit: int = -1) -> None:
        self.records: Dict[str, List[Tuple[int, int]]] = {}
        self.limit = limit

    def clear(self, limit: int = -1):
        self.records = {}
        self.limit = limit

    def add_record(self, name: str, record: Union[List[int], List[Tuple[int, int]]]):
        if isinstance(record[0], int):
            record = [(0, r) for r in record]  # type:ignore
        self.records[name] = record  # type:ignore
        return self

    def plot(self, name_to_kwargs: Callable[[str], Dict[str, str]], hue: Optional[str], style: Optional[str] = None, limit: int = -1, window: int = 8):
        pd_records: List[Dict[str, Any]] = []
        for rec_name, record in self.records.items():
            speed = extract_speed_base(record)
            if limit == -1:
                limit = self.limit
            if limit != -1:
                speed = speed[:limit]
            speed = smooth_speed(speed=speed, window=window)
            extra_args = name_to_kwargs(rec_name)
            for i, s in enumerate(speed):
                r: Dict[str, Any] = {"Seq Length": i*window, "Parallel": rec_name, "Tokens Per Second": s}
                r.update(extra_args)
                pd_records.append(r)
        data = pd.DataFrame.from_records(pd_records)
        sns.set_style("whitegrid")
        sns.set_theme(font="Times New Roman")
        sns.lineplot(data, x="Seq Length", y="Tokens Per Second", hue=hue, style=style)

    def speed_info(self, ranges: Optional[List[int]] = None):
        max_name_len = max([len(n) for n in self.records.keys()])
        for name, record in self.records.items():
            sp = compute_avg_speed(record, ranges)
            s = '\t'.join([f'{s:.2f}' for s in sp])
            print(f"{name} {' '*(max_name_len-len(name))}\t{s}")
