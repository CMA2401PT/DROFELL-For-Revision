import os
import sys
import pickle
from typing import Set, List, Tuple, Dict, Any, Optional
import plotly.graph_objects as go

from typing import Callable


def make_fig(xaxis: str, yaxis: str):
    fig = go.Figure(layout=go.Layout(
        font={
            "family": "Times New Roman",
            "size": 20,
        },
        xaxis=dict(
            title=dict(
                text=xaxis
            ),

        ),
        yaxis=dict(
            title=dict(
                text=yaxis
            ),
            # range=[0, 5500],
        ),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    ))
    return fig


def get_unit_scale(unit: str) -> Callable[[float], float]:
    if unit == "GiB":
        return lambda x: x/(2**30)
    elif unit == "MiB":
        return lambda x: x/(2**20)
    else:
        assert False, unit


def plot_lines(
    fig: go.Figure,
    lines: List[Tuple[str, List[float]]],
    stack_lines: List[Tuple[str, List[float]]],
    scale: str = "MiB"
):
    u = get_unit_scale(scale)
    last_y: Dict[int, float] = {}
    if len(stack_lines) > 0:
        for name, data in stack_lines:
            new_data: List[float] = []
            for step, y in enumerate(data):
                new_y = y+last_y.get(step, 0)
                last_y[step] = new_y
                new_data.append(new_y)
            x = [i for i in range(len(data))]
            fig.add_trace(trace=go.Scatter(x=x, y=[u(v) for v in new_data], name=name, fill='tonexty', line=dict(dash='dash')))

    for name, data in lines:
        x = [i for i in range(len(data))]
        fig.add_trace(trace=go.Scatter(x=x, y=[u(v) for v in data], name=name))
    return fig
