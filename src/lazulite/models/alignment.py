"""行级对齐的模型无关结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

TimingSource = Literal[
    "model_units",
    "acoustic_core",
    "source_offset",
    "interpolated",
    "unresolved",
]


def _validate_time(value: float | None, field_name: str) -> None:
    if value is not None and not isfinite(value):
        raise ValueError(f"{field_name} 必须为有限秒数或 None")


@dataclass(frozen=True, slots=True)
class AlignedLyricLine:
    line_id: str
    start: float | None
    end: float | None
    similarity: float | None
    score: float | None
    timing_source: TimingSource
    source_segment_ids: tuple[str, ...]
    issues: tuple[str, ...] = ()
    matched_text: str | None = None
    start_source: str | None = None
    end_source: str | None = None
    source_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.line_id:
            raise ValueError("对齐结果必须关联 line_id")
        _validate_time(self.start, "AlignedLyricLine.start")
        _validate_time(self.end, "AlignedLyricLine.end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("对齐行的开始时间不能晚于结束时间")
        for field_name in ("similarity", "score"):
            value = getattr(self, field_name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{field_name} 必须为有限数值或 None")
        if self.timing_source == "unresolved" and (self.start is not None or self.end is not None):
            raise ValueError("unresolved 行不能携带时间")


@dataclass(frozen=True, slots=True)
class AlignmentAnchor:
    line_id: str
    source_segment_id: str
    offset: float
    weight: float | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.line_id or not self.source_segment_id:
            raise ValueError("anchor 必须关联歌词行和声学分片")
        if not isfinite(self.offset):
            raise ValueError("anchor offset 必须为有限秒数")
        if self.weight is not None and not isfinite(self.weight):
            raise ValueError("anchor weight 必须为有限数值或 None")


@dataclass(frozen=True, slots=True)
class LineAlignmentResult:
    lines: tuple[AlignedLyricLine, ...]
    strategy: str
    offset_accepted: bool
    anchors: tuple[AlignmentAnchor, ...]
    diagnostics: tuple[str, ...] = ()

    coverage: dict = field(init=False)
    status: str = field(init=False)
    issue_counts: dict = field(init=False)

    def __post_init__(self):
        from collections import Counter
        complete=sum(bool(l.start is not None and l.end is not None and l.start<l.end) for l in self.lines)
        single=sum((l.start is None)!=(l.end is None) for l in self.lines)
        coverage={'total':len(self.lines),'matched':sum(bool(l.source_segment_ids) for l in self.lines),
                  'complete':complete,'single_endpoint':single,'unresolved':len(self.lines)-complete-single}
        object.__setattr__(self,'coverage',coverage)
        object.__setattr__(self,'status','complete' if self.lines and complete==len(self.lines) else 'partial' if complete or single else 'unresolved')
        object.__setattr__(self,'issue_counts',dict(Counter(issue for line in self.lines for issue in line.issues)))
