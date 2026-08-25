from plotly.colors import sample_colorscale  # type: ignore
import plotly.graph_objects as go  # type: ignore
from typing import Optional
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing import Dict, List

import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple,  Union
from numpy import std
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_avg_speed(record: List[Tuple[int, int]], ranges: Optional[List[int]] = None) -> List[float]:
    result: List[float] = []
    if ranges is None:
        ranges = [128, 1024, 2048, -1]
    for r in ranges:
        if r == -1:
            result.append(len(record)/((record[-1][1]-record[0][1])/10**9))
            result.append(len(record))
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


class Analizer:
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
            sub = '\t'.join([f'{s:.1f}' for s in sp])
            print(f"{name}{' '*(max_name_len-len(name))}\t{sub}")


@dataclass
class TimeTracer:
    time_record: List[Tuple[str, int]]

    @staticmethod
    def _record_to_str(trace: List[Tuple[str, int]]):
        start_item, start_time = trace[0]
        ds = [e[1]-s[1] for s, e in zip(trace[:-1], trace[1:])]
        max_value = (max(ds))
        _, end_time = trace[-1]
        total_time = end_time-start_time
        s = f"{start_item}"
        for (i, e), d in zip(trace[1:], ds):
            p = d/total_time
            value_ratio = d / max_value
            c = "{}"
            if value_ratio < 0.25:
                c = "\033[37m{}\033[0m"
            elif value_ratio < 0.5:
                c = "\033[32m{}\033[0m"
            elif value_ratio < 0.75:
                c = "\033[33m{}\033[0m"
            else:
                c = "\033[31m{}\033[0m"
            if p < 0.1:
                s += f"->{i}"
            else:
                s += c.format(f"-({d/(10**6):.0f}ms,{p:.0%})->")+f"{i}"
            start_time = e
        return s

    @classmethod
    def _clean_records(cls, records: Sequence[Union[List[Tuple[str, int]], 'TimeTracer']]) -> List[List[Tuple[str, int]]]:
        _records: List[List[Tuple[str, int]]] = []
        for r in records:
            if isinstance(r, TimeTracer):
                _records.append(r.time_record)
            else:
                _records.append(r)
        return _records

    @classmethod
    def gather_records_by_type(cls, records: Sequence[Union[List[Tuple[str, int]], 'TimeTracer']]):
        _records = cls._clean_records(records)
        stat: List[Tuple[str, List[Tuple[int, int]]]] = []
        start_time = _records[0][0][1]
        _last_desc = "last"
        for d, _ in _records[0]:
            stat.append((f"{_last_desc}->{d}", []))
            _last_desc = d
        for _record in _records:
            for _i, (_, t) in enumerate(_record):
                stat[_i][1].append((start_time, t-start_time))
                start_time = t
        return stat

    @classmethod
    def gathered_records_to_str(cls, stats: List[Tuple[str, List[Tuple[int, int]]]], remove_abnormal: bool = False) -> Tuple[str, str]:
        if remove_abnormal:
            stats_modified: List[Tuple[str, List[Tuple[int, int]]]] = []
            for desc, ts in stats:
                desc_total_time = sum(duration for (_, duration) in ts)
                average_time = desc_total_time/len(ts)
                desc_std = float(std(list(duration for (_, duration) in ts)))
                abnormal_thres = average_time+desc_std+10**6
                stats_modified.append((desc, [(start, (duration if duration <= abnormal_thres else int(average_time))) for (start, duration) in ts]))
            return cls.gathered_records_to_str(stats_modified, False)
        abnormal_value: str = ""
        total_time = sum(duration for (_, ts) in stats for (_, duration) in ts)
        stats_str: str = f"total: {(total_time/len(stats[0][1]))/(10**6):6.4f}ms\n"
        for desc, ts in stats:
            desc_total_time = sum(duration for (_, duration) in ts)
            desc_percentage = desc_total_time/total_time
            average_time = desc_total_time/len(ts)
            desc_std = float(std(list(duration for (_, duration) in ts)))
            abnormal_thres = average_time+desc_std+10**6
            stats_str += f"{desc:<60}:"
            if desc_percentage > 0.05:
                stats_str += f"\033[31m[!]{average_time/(10**6):6.4f}ms \t{desc_percentage:.2%}\033[0m\t"
            else:
                stats_str += f"   {average_time/(10**6):6.4f}ms \t{desc_percentage:4.2%}\t"
            if desc_std > 10**6:
                stats_str += f"\033[31m[!](±{desc_std/(10**6):6.4f})\033[0m\n"
            else:
                stats_str += f"   (±{desc_std/(10**6):6.4f})\n"
            for i, (_, t) in enumerate(ts):
                if t > abnormal_thres:
                    abnormal_value += f"abnormal: {desc} {t/(10**6):6.4f}ms @ {i}\n"
        return stats_str.removesuffix("\n"), abnormal_value

    @classmethod
    def records_to_detail_str(cls, records: Sequence[Union[List[Tuple[str, int]], 'TimeTracer']]) -> str:
        _records = cls._clean_records(records)
        start_trace = _records[0]
        _, start_time = start_trace[0]
        total_time = _records[-1][-1][1]-_records[0][0][1]
        s = f"total: {total_time/(10**6):.0f}ms\n"
        last_end = start_trace[0][1]

        sub_strs: List[Tuple[int, str, int]] = []
        for i, (_record) in enumerate(_records):
            start_time: int = _record[0][1]
            ss = ""
            if (start_time-last_end) > (10**5):
                ss += f"-({(start_time-last_end)/(10**6):.0f})->"
            ss += cls._record_to_str(_record)+"\n"
            this_end = _record[-1][1]
            sub_strs.append((i, ss, this_end-start_time))
            last_end = this_end
        last = ("", 0)
        acc: List[int] = []

        def flush(s: str):
            if len(acc) == 0:
                return s
            if len(acc) == 1:
                s += f"{acc[0]}: {last[1]/(10**6):.0f}ms ({last[1]/total_time:.1%}) \n  {last[0]}"
            else:
                s += f"{acc[0]}-{acc[-1]}: "
                s += f"\033[34m{len(acc)}x\033[0m "
                s += f"{last[1]/(10**6):.0f}ms ({last[1]/total_time:.1%}) = {last[1]*len(acc)/(10**6):.0f}ms ({last[1]*len(acc)/total_time:.1%})\n"
                s += f"  {last[0]}"
            return s
        for i, ss, d in sub_strs:
            if ss != last[0]:
                s = flush(s)
                last = (ss, d)
                acc = [i]
            else:
                acc.append(i)
        s = flush(s)
        return s

    @classmethod
    def records_to_str(cls, records: Sequence[Union[List[Tuple[str, int]], 'TimeTracer']], detail: bool = True, remove_abnormal: bool = False):
        s = ""
        if detail:
            s += cls.records_to_detail_str(records)
        stats = cls.gather_records_by_type(records)
        detail_str, abnormal_info = cls.gathered_records_to_str(stats, False)
        s += detail_str
        if remove_abnormal:
            s += "\nremove abnormal"
            detail_str, _ = cls.gathered_records_to_str(stats, True)
            s += "\n"+detail_str
        return s, abnormal_info


def gen_detail_plot_from_gathered(stat: List[Tuple[str, List[Tuple[int, int]]]]):
    records: List[Dict[str, Any]] = []
    for _type, _records in stat:
        for _step, (_start, _duration) in enumerate(_records):
            _type = _type.replace("->", "→")
            records.append({"stage": _type, "step": _step, "start": _start/(10**6), "duration": _duration/(10**6)})
    data = pd.DataFrame.from_records(records)
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="Times New Roman")
    sns.lineplot(data=data, x="start", y="duration", hue="stage")
    plt.ylim(0, 8)
    plt.ylabel("Duration (Million Second)")
    plt.xlabel("Operate")
    plt.tight_layout()


def gen_detail_plot(records: Sequence[Union[List[Tuple[str, int]], 'TimeTracer']]):
    records_by_type = TimeTracer.gather_records_by_type(records)
    gen_detail_plot_from_gathered(records_by_type)


Record = Tuple[str, int]
StepRecords = List[Record]
TaskStepsRecords = List[List[Record]]
MultiTasksStepsRecords = List[TaskStepsRecords]


def plot_timeline(
    data: MultiTasksStepsRecords,
    fig: Optional[go.Figure] = None,
    trim_events: Optional[Tuple[float, float]] = None,
    trim_time: Optional[Tuple[float, float]] = None,
    style_override: Optional[bool] = False,
    opacity_override: Optional[bool] = False,
    hidden_override: Optional[bool] = False,
    group_range_ms: float = 1000,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    trimmed_events: List[Tuple[int, str, float]] = []
    tasks_count = len(data)
    events_count = 0
    event_name_idx: Dict[str, int] = {}
    for task_i, task_records in enumerate(data):
        events: List[Tuple[str, int]] = []
        for steps_records in task_records:
            events.extend(steps_records)
        if trim_events is not None:
            start = int(len(events)*trim_events[0])
            end = int(len(events)*trim_events[1])
            events = events[start:end]
        for event_name, timestamp in events:
            if not event_name in event_name_idx:
                event_name_idx[event_name] = events_count
                events_count += 1
            trimmed_events.append((task_i, event_name, timestamp/(10**6)))
    trimmed_events = sorted(trimmed_events, key=lambda event: event[2])
    start_time = trimmed_events[0][2]
    end_time = trimmed_events[-1][2]
    if trim_time is not None:
        duration = end_time-start_time
        end_time = start_time+duration*trim_time[1]
        start_time = start_time+duration*trim_time[0]
    # trimmed_events=[(t,e,stamp) for (t,e,stamp) in trimmed_events if stamp>=start_time and stamp<=end_time]
    events_count = len(event_name_idx)

    mode = "event_group"

    if mode == "task":
        tasks_event: List[List[Tuple[str, float]]] = [[]]*tasks_count
        for task_i, event_name, event_time in trimmed_events:
            tasks_event[task_i].append((event_name, event_time))
        for (task_i, task_event) in enumerate(tasks_event):
            fig.add_trace(go.Scatter(
                x=[t for (_, t) in task_event],
                y=[task_i]*len(task_event),
                text=[e for (e, _) in task_event],
            ))
    if mode == "event_group":
        event_groups: List[Tuple[str, List[Tuple[int, float]]]] = []
        event_group_ptr: Dict[str, List[List[Tuple[int, float]]]] = {event_name: [] for event_name in event_name_idx}
        for task_i, event_name, event_time in trimmed_events:
            ptr: List[List[Tuple[int, float]]] = event_group_ptr[event_name]
            gptr: Optional[List[Tuple[int, float]]] = None
            gptr_i: int = 0
            for _i, _p in enumerate(ptr):
                exist = False
                _min, _max = math.inf, -math.inf
                for (_ti, _t) in _p:
                    _min = min(_min, _t)
                    _max = max(_max, _t)
                    if _ti == task_i:
                        exist = True
                        break
                if not exist:
                    gptr, gptr_i = _p, _i
                    break
                    # if event_time>_min-group_range_ms and event_time<_max+group_range_ms:

            if gptr is None:
                gptr, gptr_i = [], len(ptr)
                event_groups.append((event_name, gptr))
                ptr.append(gptr)
            gptr.append((task_i, event_time))
            if len(gptr) == tasks_count:
                ptr = [_p for i, _p in enumerate(ptr) if i != gptr_i]
                event_group_ptr[event_name] = ptr
        color_mapping: Dict[str, Any] = {}
        base_color = ("#00ADB5", "#EEEEEE")
        if style_override:
            for event_name in event_name_idx:
                if "exec" in event_name:
                    paired_event_type = event_name.replace("exec", "")
                    color_mapping[paired_event_type] = None
                elif "done" in event_name:
                    paired_event_type = event_name.replace("done", "")
                    color_mapping[paired_event_type] = None
            step = 1/len(color_mapping)
            cs = sample_colorscale(base_color, [i*step for i in range(len(color_mapping))])
            color_mapping = {k: cs[i] for (i, k) in enumerate(color_mapping)}

        for event_name, yxs in event_groups:
            if hidden_override:
                if "exec" in event_name or "done" in event_name or "leave" in event_name or "recv" in event_name:
                    pass
                else:
                    continue
            yxs = [(t, stamp) for (t, stamp) in yxs if stamp >= start_time and stamp <= end_time]
            if len(yxs) == 0:
                continue
            opacity = 1
            color = None
            dash = None
            if opacity_override:
                opacity = 0.3
                if "exec" in event_name or "done" in event_name or "leave" in event_name or "recv" in event_name:
                    opacity = 1
            if style_override:
                if "exec" in event_name or "done" in event_name or "leave" in event_name or "recv" in event_name:
                    if "exec" in event_name:
                        paired_event_type = event_name.replace("exec", "")
                        color = color_mapping[paired_event_type]
                    elif "done" in event_name:
                        paired_event_type = event_name.replace("done", "")
                        color = color_mapping[paired_event_type]
                        dash = "dash"
                        opacity = 0.7
                    elif "leave" in event_name:
                        color = "#252A34"
                    elif "recv" in event_name:
                        color = "#FF2E63"
            fig.add_trace(go.Scatter(
                opacity=opacity,
                y=[y for (y, _) in yxs],
                x=[x for (_, x) in yxs],
                text=[event_name]*len(yxs),
                name=event_name,
                line=dict(
                    color=color if color is not None else None,
                    dash=dash if dash is not None else None,
                )
            ))

    # style all the traces
    fig.update_traces(
        hoverinfo="x+text",
        line={"width": 2},
        marker={"size": 6},
        mode="lines+markers",
        # showlegend=False
    )

    # Update axes
    fig.update_layout(
        xaxis=dict(
            autorange=True,
            rangeslider=dict(
                autorange=True,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="SubTask"
            )
        ),
        font={
            "family": "Times New Roman"
        },
    )

    # Update layout
    fig.update_layout(
        dragmode="zoom",
        hovermode="x",
        height=600,
        template="plotly_white",
    )
    return fig


def parallel_amount(tasks: MultiTasksStepsRecords):
    overall_execute_time: int = 0
    timeline_events: List[Tuple[int, str, int]] = []
    for (task_i, task_records) in enumerate(tasks):
        for steps in task_records:
            for event_name, time_stamp in steps:
                timeline_events.append((task_i, event_name, time_stamp))
    any_execute_time = 0
    overall_execute_time: int = 0
    timeline_events = sorted(timeline_events, key=lambda e: e[2])

    any_exec_start_time: Optional[int] = None
    current_exec_tasks: Set[int] = set()
    for (task_i, event_name, time_stamp) in timeline_events:
        if "exec" in event_name:
            current_exec_tasks.add(task_i)
            if any_exec_start_time is not None:
                duration = time_stamp-any_exec_start_time
                any_execute_time += duration
                overall_execute_time += duration*len(current_exec_tasks)
            any_exec_start_time = time_stamp
        if "done" in event_name:
            current_exec_tasks.remove(task_i)
            if any_exec_start_time is not None:
                duration = time_stamp-any_exec_start_time
                any_execute_time += duration
                overall_execute_time += duration*(len(current_exec_tasks)+1)
            if len(current_exec_tasks) == 0:
                any_exec_start_time = None
            else:
                any_exec_start_time = time_stamp
    print(f"parallel amount: {overall_execute_time/any_execute_time:.2%}")
    start_time = timeline_events[0][2]
    end_time = timeline_events[-1][2]
    print(f"busy percentage: {(any_execute_time/(end_time-start_time)):.2%}")


def blocking_percentage(tasks: MultiTasksStepsRecords):
    overall_duration = 0
    overall_blocking = 0
    for (task_i, task_records) in enumerate(tasks):
        start_time = task_records[0][0][1]
        end_time = task_records[-1][-1][1]
        total_time = end_time-start_time
        blocking_time = 0
        block_start_time: Optional[int] = None
        for steps in task_records:
            for event_name, time_stamp in steps:
                if "done" in event_name:
                    block_start_time = time_stamp
                elif "leave" in event_name and block_start_time is not None:
                    duration = time_stamp-block_start_time
                    block_start_time = None
                    blocking_time += duration
        overall_duration += total_time
        overall_blocking += blocking_time
        print(f"task {task_i} blocking percentage: {blocking_time/total_time:.2%}")
    print(f"overall blocking percentage: {overall_blocking/overall_duration:.2%}")
