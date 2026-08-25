import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple


@dataclass
class StepSize:
    steps_in_each_update: int = 1
    step_batch_size: int = 4

    def samples_to_steps(self, samples: int):
        trained_steps = (samples+self.step_batch_size-1)//self.step_batch_size
        updates_count = trained_steps//self.steps_in_each_update
        return trained_steps, updates_count


@dataclass
class TrainStatus:
    step_size: StepSize
    history_loss: List[float] = field(default_factory=list)
    history_tokens: List[int] = field(default_factory=list)

    @property
    def trained_steps(self):
        return len(self.history_loss)


@dataclass
class TrainStatistics:
    status: TrainStatus
    time_window: List[float]
    tokens_window: List[int]
    total_samples: int
    total_tokens: int
    average_speed: float = 0
    average_tps: float = 0

    @property
    def step_size(self):
        return self.status.step_size

    @property
    def trained_samples(self):
        return self.status.trained_steps*self.step_size.step_batch_size

    @property
    def train_progress(self):
        train_steps, train_updates = self.step_size.samples_to_steps(self.trained_samples)
        total_steps, total_updates = self.step_size.samples_to_steps(self.total_samples)
        return (self.trained_samples, train_steps, train_updates), (self.total_samples, total_steps, total_updates)

    @property
    def current_speed(self):
        if len(self.time_window) < 2:
            return 0, 0, 0, 0
        delta_time = self.time_window[-1]-self.time_window[0]
        step_count = len(self.time_window)-1
        per_step_speed = step_count/delta_time
        per_sample_speed = per_step_speed*self.step_size.step_batch_size
        per_update_speed = per_step_speed/self.step_size.steps_in_each_update
        return per_sample_speed, per_step_speed, per_update_speed, sum(self.tokens_window)/delta_time

    @property
    def estimated_time(self) -> str:
        per_sample_speed, _, _, _ = self.current_speed
        if per_sample_speed == 0:
            return "collecting"
        seconds_remain = (self.total_samples-self.trained_samples)/per_sample_speed
        t = ""
        if seconds_remain > 3600:
            t += f"{seconds_remain//3600}h"
            seconds_remain %= 3600
        if seconds_remain > 60:
            t += f"{seconds_remain//60}m"
            seconds_remain %= 60
        t += f"{int(seconds_remain)}s"
        return t

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        (trained_samples, train_steps, train_updates), (total_samples, total_steps, total_updates) = self.train_progress
        per_sample_speed, _, _, tps = self.current_speed
        info = \
            f"{train_updates}/{total_updates}, {train_steps}/{total_steps}, {trained_samples}" +\
            f"+(curr={per_sample_speed:.2f}samples/s,avg={self.average_speed:.2f}samples/s)" +\
            (f"+(curr={tps:.2f}t/s,avg={self.average_tps:.2f}t/s)" if tps != 0 else "") +\
            f"/{total_samples}|"
        info += f" {self.estimated_time}"
        info += f" loss: {self.status.history_loss[-1]:.2f}"
        return info


class TrainStatusMaintainer:
    def __init__(self,
                 step_size: StepSize,
                 total_samples: int,
                 status: Optional[TrainStatus] = None,
                 events: Optional[Dict[str, int]] = None,
                 prefix: str = "") -> None:
        self.prefix = prefix
        if status is None:
            status = TrainStatus(step_size,  [])
        else:
            assert status.step_size == step_size
        self.statistic = TrainStatistics(
            status=status,
            time_window=[],
            total_samples=total_samples,
            tokens_window=[],
            total_tokens=0,
            average_tps=0,
        )
        self.start_time_record: List[float] = []
        self.time_record: List[float] = []
        self.ids: List[str] = []
        if events is None:
            events = {}
        events["update"] = step_size.steps_in_each_update
        self.events = events
        self.events_history: List[Tuple[int, str, Any]] = []
        self.start_time: float = 0
        self.current_samples: int = 0
        self.current_tokens: int = 0
        self.step_size = step_size

    @property
    def steps(self):
        return self.statistic.status.trained_steps

    def record_event(self, type: str, data: Any):
        self.events_history.append((self.steps, type, data))

    def dump_info(self) -> Dict[str, Any]:
        return {
            "history": self.statistic.status.history_loss,
            "tokens": self.statistic.status.history_tokens,
            "samples_in_each_step": self.statistic.status.step_size.step_batch_size,
            "steps_in_each_update": self.statistic.status.step_size.steps_in_each_update,
            "times": self.time_record,
            "start_times": self.start_time_record,
            "ids": self.ids,
            "events": self.events_history,
        }

    def dump(self, name: str):
        with open(name, "w") as fp:
            json.dump(self.dump_info(), fp)

    def should(self, event: str):
        if not event in self.events:
            return
        period = self.events[event]
        return self.steps % period == period-1

    def update_loss(self, loss: float, tokens: int = 0, start_time: float = 0, batch_id: str = ""):
        self.current_samples += self.step_size.step_batch_size
        self.current_tokens += tokens
        self.statistic.status.history_loss.append(loss)
        self.statistic.status.history_tokens.append(tokens)
        t: float = time.time()
        if self.start_time == 0:
            self.start_time = start_time if start_time != 0 else t
            self.current_samples = 0
        else:
            avg_speed = self.current_samples/(t-self.start_time)
            self.statistic.average_speed = avg_speed
            avg_tps = self.current_tokens/(t-self.start_time)
            self.statistic.average_tps = avg_tps
        self.time_record.append(t)
        self.start_time_record.append(start_time)
        self.ids.append(batch_id)
        self.statistic.time_window.append(t)
        self.statistic.tokens_window.append(tokens)
        if len(self.statistic.time_window) > 5:
            self.statistic.time_window = self.statistic.time_window[1:]
            self.statistic.tokens_window = self.statistic.tokens_window[1:]
            assert len(self.statistic.time_window) == len(self.statistic.tokens_window)
        return (self.prefix+" " if self.prefix != "" else "")+self.statistic.__str__()
