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

    @property
    def trained_steps(self):
        return len(self.history_loss)


@dataclass
class TrainStatistics:
    status: TrainStatus
    time_window: List[float]
    total_samples: int
    average_speed: float = 0

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
        if len(self.time_window)-1 < self.step_size.steps_in_each_update:
            return 0, 0, 0
        delta_time = self.time_window[-1]-self.time_window[0]
        step_count = len(self.time_window)-1
        per_step_speed = step_count/delta_time
        per_sample_speed = per_step_speed*self.step_size.step_batch_size
        per_update_speed = per_step_speed/self.step_size.steps_in_each_update
        return per_sample_speed, per_step_speed, per_update_speed

    @property
    def estimated_time(self) -> str:
        per_sample_speed, _, _ = self.current_speed
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
        per_sample_speed, _, _ = self.current_speed
        info = f"{train_updates}/{total_updates}, {train_steps}/{total_steps}, {trained_samples}+({per_sample_speed:.2f},{self.average_speed:.2f})/{total_samples}|"
        info += f" {self.estimated_time}"
        info += f" loss: {self.status.history_loss[-1]:.2f}"
        return info


class TrainStatusMaintainer:
    def __init__(self, step_size: StepSize, total_samples: int, status: Optional[TrainStatus] = None, events: Optional[Dict[str, int]] = None) -> None:
        if status is None:
            status = TrainStatus(step_size,  [])
        else:
            assert status.step_size == step_size
        self.statistic = TrainStatistics(
            status=status,
            time_window=[],
            total_samples=total_samples
        )
        self.time_record: List[float] = []
        if events is None:
            events = {}
        events["update"] = step_size.steps_in_each_update
        self.events = events
        self.events_history: List[Tuple[int, str, Any]] = []
        self.start_time: float = 0
        self.current_samples: int = 0
        self.step_size = step_size

    @property
    def steps(self):
        return self.statistic.status.trained_steps

    def record_event(self, type: str, data: Any):
        self.events_history.append((self.steps, type, data))

    def dump(self, name: str):
        with open(name, "w") as fp:
            json.dump({
                "history": self.statistic.status.history_loss,
                "samples_in_each_step": self.statistic.status.step_size.step_batch_size,
                "steps_in_each_update": self.statistic.status.step_size.steps_in_each_update,
                "times": self.time_record,
                "events": self.events_history,
            }, fp)

    def should(self, event: str):
        if not event in self.events:
            return
        period = self.events[event]
        return self.steps % period == period-1

    def update_loss(self, loss: float):
        self.current_samples += self.step_size.step_batch_size
        self.statistic.status.history_loss.append(loss)
        t = time.time()
        if self.start_time == 0:
            self.start_time = t
            self.current_samples = 0
        else:
            avg_speed = self.current_samples/(t-self.start_time)
            self.statistic.average_speed = avg_speed
        self.time_record.append(t)
        self.statistic.time_window.append(t)
        if len(self.statistic.time_window) > 5:
            self.statistic.time_window = self.statistic.time_window[1:]
        return self.statistic.__str__()
