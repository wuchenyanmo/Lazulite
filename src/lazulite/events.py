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
