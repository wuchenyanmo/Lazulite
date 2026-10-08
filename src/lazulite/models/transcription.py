"""模型无关的转写结果与时间证据。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

TimedTextGranularity = Literal["subword", "word", "character", "segment"]
TimedTextSource = Literal["model", "alignment"]


def _validate_time(value: float | None, field_name: str) -> None:
    if value is not None and not isfinite(value):
        raise ValueError(f"{field_name} 必须为有限秒数或 None")


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    model_id: str
    revision: str
    runtime_version: str
    precision: str

    def __post_init__(self) -> None:
        if not all(
            (self.name, self.model_id, self.revision, self.runtime_version, self.precision)
        ):
            raise ValueError("后端身份字段均不能为空")


@dataclass(frozen=True, slots=True)
class TimedText:
    text: str
    start: float | None
    end: float | None
    granularity: TimedTextGranularity
    source: TimedTextSource
    start_source: str | None = None
    end_source: str | None = None
    generation_steps: tuple[int, ...] = ()
    probability: ProbabilitySummary | None = None

    def __post_init__(self) -> None:
        _validate_time(self.start, "TimedText.start")
        _validate_time(self.end, "TimedText.end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("TimedText 的开始时间不能晚于结束时间")


@dataclass(frozen=True, slots=True)
class RequestWindow:
    request_id: str
    source_segment_id: str
    start: float
    end: float
    core_start: float | None
    core_end: float | None
    sequence: int = 0
    section_id: str | None = None
    presence: float | None = None
    off_center_risk: float | None = None

    @property
    def id(self):
        return self.source_segment_id

    def __post_init__(self):
        if not self.request_id or not self.source_segment_id or not 0 <= self.start < self.end or not isfinite(self.end):
            raise ValueError('invalid transcription window')
        if (self.core_start is None) != (self.core_end is None):
            raise ValueError('core intersection must have both endpoints or neither')
        if self.core_start is not None and not self.start <= self.core_start < self.core_end <= self.end:
            raise ValueError('core outside request window')


@dataclass(frozen=True, slots=True)
class ProbabilitySummary:
    mean: float
    minimum: float
    low_fraction: float
    count: int
    threshold: float = .35
    definition: str = 'lexical_generation_probabilities-v1'

    def __post_init__(self):
        if self.count < 1 or not all(isfinite(x) and 0 <= x <= 1 for x in (self.mean,self.minimum,self.low_fraction,self.threshold)):
            raise ValueError('invalid probability summary')


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    source_segment_id: str
    text: str
    language: str | None
    start: float | None
    end: float | None
    confidence: float | None
    confidence_kind: str | None
    timed_text: tuple[TimedText, ...]
    backend_metadata: dict[str, object]
    request_window: RequestWindow | None = None
    probability: ProbabilitySummary | None = None
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_segment_id:
            raise ValueError("转写结果必须关联 source_segment_id")
        _validate_time(self.start, "TranscriptionSegment.start")
        _validate_time(self.end, "TranscriptionSegment.end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("转写片段的开始时间不能晚于结束时间")
        if self.confidence is not None:
            if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
                raise ValueError("confidence 必须在 [0, 1] 内或为 None")
        if self.confidence is None and self.confidence_kind is not None:
            raise ValueError("没有 confidence 时不能声明 confidence_kind")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    backend: BackendInfo
    segments: tuple[TranscriptionSegment, ...]


def transcription_segment_from_dict(value):
    """Read only the current schema's complete evidence; never upgrade legacy cache rows."""
    return TranscriptionSegment(**{**value,
        'request_window':RequestWindow(**value['request_window']) if value['request_window'] is not None else None,
        'probability':ProbabilitySummary(**value['probability']) if value['probability'] is not None else None,
        'issues':tuple(value['issues']),
        'timed_text':tuple(TimedText(**{**u,'generation_steps':tuple(u['generation_steps']),
            'probability':ProbabilitySummary(**u['probability']) if u['probability'] is not None else None}) for u in value['timed_text'])})
