"""库层的轻量进度事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

EventKind = Literal["started", "progress", "warning", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    kind: EventKind
    completed: int | None = None
    total: int | None = None
    message: str | None = None


EventSink = Callable[[ProgressEvent], None]


class Cancelled(RuntimeError):
    """The caller cancelled; no final success result is published."""


def checkpoint(cancel=None, event_sink=None, stage=None, completed=None, total=None):
    if cancel is not None and cancel():
        raise Cancelled('alignment cancelled')
    if event_sink is not None and stage is not None:
        event_sink(ProgressEvent(stage, 'progress', completed, total))
