import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


@dataclass
class InstanceData:
    history: List[float]
    tokens: List[int]
    samples_in_each_step: int
    steps_in_each_update: int
    times: List[float]
    start_times: List[float]
    ids: List[str]
    events: List[Tuple[int, str, Any]]

    @property
    def minibatches(self):
        return len(self.times)

    @property
    def start_time_valid(self):
        return self.start_times[0] != 0

    @property
    def start_time(self):
        if self.start_time_valid:
            return self.start_times[0]
        else:
            return self.times[0]

    @property
    def total_samples(self):
        if self.start_time_valid:
            return self.minibatches*self.samples_in_each_step
        else:
            return (self.minibatches-1)*self.samples_in_each_step

    @property
    def total_tokens(self):
        if self.start_time_valid:
            return sum(self.tokens)
        else:
            return sum(self.tokens[1:])

    @property
    def total_time(self):
        if self.start_time_valid:
            return (self.times[-1]-self.start_times[0])
        else:
            return (self.times[-1]-self.times[0])

    @property
    def sample_per_second(self):
        return self.total_samples/self.total_time

    @property
    def token_per_second(self):
        return self.total_tokens/self.total_time

    @property
    def time_block(self):
        blocks: List[Tuple[str, float, float]] = []
        assert self.start_time_valid
        for _id, _s, _e in zip(self.ids, self.start_times, self.times):
            blocks.append((_id, _s, _e))

        return blocks

    def speed(self, smooth: int = 4):
        speed: List[float] = []
        for i in range(len(self.times)-smooth):
            d = self.times[i+smooth]-self.times[i]
            speed.append((smooth/d)*self.samples_in_each_step)
        return speed

    def tps(self, smooth: int = 4):
        speed: List[float] = []
        for i in range(len(self.times)-smooth):
            d = self.times[i+smooth]-self.times[i]
            t = sum(self.tokens[i:i+smooth])
            speed.append((t/d))
        return speed

    def pd_records(self, speed_smooth: int = 4, dump: Tuple[str, ...] = ("loss", "speed", "tps"), **extra: Any):
        records: List[Dict[str, Any]] = []
        if "loss" in dump:
            for i, h in enumerate(self.history):
                records.append({"time": self.times[i]-self.start_time, "Data": "Loss", "value": h, **extra})
        if "speed" in dump:
            for i, s in enumerate(self.speed(speed_smooth)):
                records.append({"time": self.times[i]-self.start_time, "Data": "Speed (minibatch/s)", "value": s,  **extra})
        if "tps" in dump:
            for i, s in enumerate(self.tps(speed_smooth)):
                records.append({"time": self.times[i]-self.start_time, "Data": "TPS (t/s)", "value": s,  **extra})
        return records

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, list) and k != "events":
                assert len(v) == self.minibatches


def plot_speed_and_loss(records: List[Dict[str, int]]):
    data = pd.DataFrame.from_records(records)
    sns.set_style("darkgrid")
    sns.set_theme(font="Times New Roman")
    # fig = plt.figure(figsize=(10, 6), dpi=330)

    g = sns.FacetGrid(
        data=data,
        col="Data",
        hue="Instance",
        sharey=False,
        # col_wrap=2,
        height=3,
        aspect=1.5,
    )
    g.figure.set_dpi(330)
    g.map(sns.lineplot, "time", "value").add_legend()
    g.set_axis_labels("Time", "Value")

    return g.figure


def plot_speed_and_loss_interact(records: List[Dict[str, int]]):
    data = pd.DataFrame.from_records(records)
    data_types = data["Data"].unique()

    fig = make_subplots(
        rows=len(data_types),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=data_types,
    )

    for i, data_name in enumerate(data_types, start=1):
        df_sub = data[data["Data"] == data_name]

        for instance_name, df in df_sub.groupby("Instance"):
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["value"],
                    mode="lines",
                    name=f"{data_name} ({instance_name})",
                    legendgroup=f"{data_name}-{instance_name}",
                ),
                row=i,
                col=1,
            )

    fig.update_layout(height=250 * len(data_types), width=900)
    return fig


@dataclass
class OperateRecord:
    idle_time: int
    begin_time: int
    delta_time: int
    extra_async_delta_time: int
    description: List[str]

    @property
    def start_time_second(self):
        return self.begin_time/1e9

    @property
    def end_time_second(self):
        return (self.begin_time+self.delta_time+self.extra_async_delta_time)/1e9

    @property
    def name(self):
        return self.description[0]


def plot_time_line(
    time_lines: List[Tuple[str, str, float, float]],
    group_fn: Callable[[str], str],
    priority: List[str],
    color_mapping: Dict[str, str],
    default_color: str = "blue",
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
):
    priority = priority[::-1]
    lines: Dict[str, List[Tuple[str, float, float]]] = {}
    types: Set[str] = set()

    obj_height = 0.8
    line_height = 1
    margin = 0.5
    gs, ge = math.inf, 0
    for typ, act, s, e in time_lines:
        gs, ge = min(s, gs), max(ge, e)
        types.add(act)
        lines.setdefault(typ, []).append((act, s, e))

    lines = {
        k: [
            (act, s-gs, e-gs) for (act, s, e) in vs
            if (trim_start is None or s-gs > trim_start)
            and (trim_end is None or s-gs < trim_end)
        ] for k, vs in lines.items()
    }
    gs, ge = 0, ge-gs
    if trim_start:
        gs = trim_start
    if trim_end:
        ge = trim_end

    groups = list(set([group_fn(g) for g in lines.keys()]))
    assert sorted(groups) == sorted(priority), f"{groups},{priority}"
    group_lines: Dict[str, List[str]] = {}
    for k in lines.keys():
        g = group_fn(k)
        group_lines.setdefault(g, []).append(k)
    group_lines = {k: sorted(v) for k, v in group_lines.items()}
    y = 0
    line_ys: Dict[str, float] = {}
    for gi, g in enumerate(priority):
        ks = group_lines[g]
        for k in ks:
            line_ys[k] = y*line_height+gi*margin
            y += 1

    fig = go.Figure()
    ytickvals = []
    yticktext = []
    legend_shown: Set[str] = set()

    def add_rect(x: Tuple[float, float], y: float, color: Any, name: Optional[str], text: Optional[str], **kwargs: Any):
        x0, x1 = x
        y0, y1 = y, y+obj_height
        showlegend = False
        if name is not None and not name in legend_shown:
            legend_shown.add(name)
            showlegend = True
        kwargs.update(dict(
            x=[x0, x0, x1, x1, x0],
            y=[y0, y1, y1, y0, y0],
            fill="toself",
            mode='lines',
            name=name,
            text=text,
            line=dict(color=color),
            showlegend=showlegend
        ))
        fig.add_trace(go.Scatter(**kwargs))

    def add_line(y: float, name: str, color: Any):
        ytickvals.append(y+0.5*line_height)
        yticktext.append(name)
        fig.add_trace(go.Scatter(dict(
            x=[0, 0, ge, ge, 0],
            y=[y, y+line_height, y+line_height, y, y],
            fill="toself",
            mode='lines',
            line=dict(color=color),
            showlegend=False
        )))
    for line_name, blocks in lines.items():
        y = line_ys[line_name]
        add_line(y, line_name, "lightgrey")
        for act, s, e in blocks:
            c = color_mapping.get(act, default_color)
            add_rect((s, e), y, c, act, act)

    fig.update_layout(
        xaxis=dict(
            # autorange=True,
            rangeslider=dict(
                range=(gs, ge),
            ),
            tick0=0,
            title=dict(
                text="Time (Second)"
            ),
            range=(gs, ge)
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=ytickvals,
            ticktext=yticktext,
        ),
        font={
            "family": "Times New Roman",
            "size": 20,
        },
        height=600,
    )
    return fig


if __name__ == "__main__":
    import argparse
    import json
    import os

    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    output_dir = args.output

    with open(os.path.join(output_dir, "results.json"), "r") as f:
        instance_data = [
            InstanceData(**i)
            for i in json.load(f)
        ]

    for i, instance in enumerate(instance_data):
        name = f"instance-{i}:\t"
        if i == len(instance_data)-1:
            name = "overall:  \t"
        print(f"{name} {instance.sample_per_second:5.2f} samples/s \t{instance.token_per_second:5.2f} t/s")

    total_records = [
        r
        for i, instance in enumerate(instance_data)
        for r in instance.pd_records(
            smooth=1,
            speed_smooth=4,
            dump=("speed", "tps"),
            Instance=f"instance-{i}" if i != len(instance_data) else "overall",
        )
    ]

    fig = plot_speed_and_loss(total_records)
    fig.savefig(os.path.join(output_dir, "speed.png"), dpi=330)

    fig = plot_speed_and_loss_interact(total_records)
    fig.write_html(os.path.join(output_dir, "speed.html"))

    with open(os.path.join(output_dir, "brain.json"), "r") as f:
        busys = {cast(str, k): [OperateRecord(**v) for (name, v) in vs] for k, vs in json.load(f).items()}

    time_lines: List[Tuple[str, str, float, float]] = []
    instances: Set[str] = set()
    for instance_i, instance in enumerate(instance_data[:-1]):
        blocks = instance.time_block
        last_batch = int(blocks[-1][0].split("-")[0])
        batch_size = round(len(blocks)/last_batch)
        for (_id, _s, _e) in blocks:
            batch_i, minibatch_i = [int(v)for v in _id.split("-")]
            line_name = f"{instance_i}-{minibatch_i%batch_size}"
            instances.add(f"{instance_i}")
            time_lines.append((line_name, "sample", _s, _e))

    for dev_i, records in busys.items():
        for record in records:
            time_lines.append((f"dev-{dev_i}", record.name, record.start_time_second, record.end_time_second))

    overall_start = min(vs[0].start_time_second for vs in busys.values())
    dev0_busys = list(busys.values())[0]
    operates = len(dev0_busys)
    possible_start = dev0_busys[operates//3].start_time_second
    possible_end = dev0_busys[min(operates-1, operates//3+1000)].start_time_second

    fig = plot_time_line(
        time_lines=time_lines,
        group_fn=lambda s: s.split("-")[0],
        priority=list(sorted(list(instances)))+["dev"],
        color_mapping={
            "forward": "green",
            "backward": "red",
            "train_head": "brown",
            "train_tail": "brown",
            "loss": "brown",
            "weight_update": "goldenrod"
        },
        trim_start=possible_start-overall_start,
        trim_end=possible_end-overall_start
    )
    fig.write_html(os.path.join(output_dir, "detail.html"))
