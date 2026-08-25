import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
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
        name = f"client-{i}:\t"
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

    with open(os.path.join(output_dir, "traces.json"), "r") as f:
        data = json.load(f)

    Record = Tuple[str, float]
    StepRecords = List[Record]
    TaskStepsRecords = List[List[Record]]
    MultiTasksStepsRecords = Dict[str, TaskStepsRecords]
    data: MultiTasksStepsRecords = {
        name: [
            [
                (name, stamp/1e9)
                for (name, stamp) in step
            ]
            for step in task
        ]
        for name, task in data.items()
    }
    print(f"Parallel Amount {len(data)}")

    blocks: Dict[str, List[Tuple[str, float, float]]] = {}
    for task_i, task_records in data.items():
        _blocks: List[Tuple[str, float, float]] = []
        for steps_records in task_records:
            start, start_time = steps_records[0]
            for (ev, time_stamp) in steps_records[1:]:
                action_name = f"{start}->{ev}"
                _blocks.append((action_name, start_time, time_stamp))
                start, start_time = ev, time_stamp
        blocks[task_i] = _blocks

        event_names: Set[str] = set()
        for _blocks in blocks.values():
            for action, _, _ in _blocks:
                event_names.add(action)

        async_ev = ("async", "goldenrod")
        forward_ev = ("forward", "green")
        backward_ev = ("backward", "red")
        net_ev = ("net", "black")
        # net_extra_ev=("net","brown")
        idle_ev = ("idle", "goldenrod")
        update_ev = ("update", "blue")
        swap_ev = ("swap", "brown")
        dump_ev = ("dump", "pink")
        mapping: Dict[str, Tuple[str, str]] = {
            'root->finish': idle_ev,
            'start->leave': idle_ev,
            'wait_head->exec_head': idle_ev,
            'wait_tail->exec_tail': idle_ev,
            'dealer-recv->wait_head': idle_ev,
            'wait_forward->exec_forward': idle_ev,
            'wait_backward->exec_backward': idle_ev,
            'async_forward->wait_tail': idle_ev,
            'async_forward->wait_forward': idle_ev,
            'async_head->wait_forward': idle_ev,
            'async_backward->leave': idle_ev,
            'async_backward->wait_backward': idle_ev,
            'async_tail->wait_backward': idle_ev,
            'async_forward->leave': idle_ev,
            'dealer-recv->wait_forward': idle_ev,
            'dealer-recv->wait_backward': idle_ev,

            'leave->root': idle_ev,
            'root->leave': idle_ev,

            'leave->dealer-recv': net_ev,


            'done_head->async_head': async_ev,
            'done_tail->async_tail': async_ev,
            'done_forward->async_forward': async_ev,
            'done_backward->async_backward': async_ev,

            'exec_head->done_head': forward_ev,
            'exec_tail->done_tail': forward_ev,
            'exec_forward->done_forward': forward_ev,
            'exec_backward->done_backward': backward_ev,

            'forward': forward_ev,
            'backward': backward_ev,
            'train_head': forward_ev,
            'train_tail': forward_ev,
            'weight_update': update_ev,
            'swapout': swap_ev,
            'dump_state': dump_ev,
        }

        time_lines: List[Tuple[str, str, float, float]] = []
        color_mapping: Dict[str, str] = {}
        _instances: Set[str] = set()
        for minibatch_i, instance_blocks in blocks.items():
            name = f"client-{minibatch_i}"
            instance_i = "client-"+("-".join(minibatch_i.split('-')[:-1]))
            _instances.add(instance_i)
            for (act, s, e) in instance_blocks:
                if mapping[act] == idle_ev:
                    continue
                color_mapping[mapping[act][0]] = mapping[act][1]
                time_lines.append((name, mapping[act][0], s, e))
        instances = sorted(list(_instances))

        from typing import cast

        def simplify_dealer_name(k: str):
            if k.startswith("T"):
                return k.split('-', 1)[1].replace("dealer", "device")
            return k.replace("dealer", "device")

        with open(os.path.join(output_dir, "brains.json"), "r") as f:
            busys = {simplify_dealer_name(cast(str, k)): [OperateRecord(**v) for (name, v) in vs] for k, vs in json.load(f).items()}

        for dealer_name, records in busys.items():
            for record in records:
                if record.name in ["remove_decoders_train_binding", "create_train_binding"]:
                    continue
                action, color = mapping[record.name]
                color_mapping[action] = color
                time_lines.append((dealer_name, action, record.start_time_second, record.end_time_second))

        def group_fn(name: str):
            if name.startswith("dealer") or name.startswith("device"):
                return "device"
            instance_i = "-".join(name.split('-')[:-1])
            return instance_i

        timeline0 = list(blocks.values())[0]
        operates = len(timeline0)
        possible_begin = timeline0[0][1]
        possible_start = timeline0[operates//3][1]
        possible_end = timeline0[min(operates-1, operates//3+100)][1]

        fig = plot_time_line(
            time_lines=time_lines,
            group_fn=group_fn,
            priority=instances+["device"],
            color_mapping=color_mapping,
            trim_start=possible_start-possible_begin,
            trim_end=possible_end-possible_begin
        )

        fig.write_html(os.path.join(output_dir, "detail.html"))
