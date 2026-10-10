"""音频与声学分片的数据约定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _require_finite_time(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} 必须是有限秒数")


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """供歌词搜索使用的轻量文件元数据，不携带封面或完整 tag dump。"""

    path: Path
    title: str | None
    artist: str | None
    album: str | None
    duration: float

    def __post_init__(self) -> None:
        _require_finite_time(self.duration, "duration")
        if self.duration < 0:
            raise ValueError("音频时长不能为负数")


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """解码后的音频，布局始终为 ``float32[channels, samples]``。"""

    samples: NDArray[np.float32]
    sample_rate: int
    source_id: str = "audio"

    def __post_init__(self) -> None:
        if self.samples.dtype != np.float32:
            raise ValueError("音频数组必须使用 float32")
        if self.samples.ndim != 2:
            raise ValueError("音频数组必须为 [channels, samples] 两维布局")
        if self.samples.shape[0] < 1:
            raise ValueError("音频至少需要一个声道")
        if not np.isfinite(self.samples).all():
            raise ValueError("PCM samples must be finite")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("采样率必须为正数")
        if not self.source_id:
            raise ValueError("音频必须具有非空 source_id")

    @property
    def duration(self) -> float:
        """音频时长；第一个解码样本固定对应整曲时间零点。"""
        return self.samples.shape[1] / self.sample_rate


@dataclass(frozen=True, slots=True)
class VocalSegment:
    """带 padding 窗口与核心人声区间的声学片段。"""

    id: str
    section_id: str | None
    start: float
    end: float
    core_start: float
    core_end: float
    presence: float | None
    off_center_risk: float | None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("声学分片 id 不能为空")
        for field_name in ("start", "end", "core_start", "core_end"):
            _require_finite_time(getattr(self, field_name), field_name)
        if not self.start < self.end:
            raise ValueError("声学分片必须满足 start < end")
        if not (self.start <= self.core_start < self.core_end <= self.end):
            raise ValueError("核心人声区间必须落在分片窗口内")
        for field_name in ("presence", "off_center_risk"):
            value = getattr(self, field_name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{field_name} 必须为有限数值或 None")


@dataclass(frozen=True, slots=True)
class VocalAnalysis:
    """一次声学分析的结果；数组留在音频层，不复制进该结果。"""

    source_id: str
    segments: tuple[VocalSegment, ...]
    diagnostics: tuple[str, ...] = ()
    is_vocal: bool | None = None
    vocal_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("声学分析必须具有非空 source_id")
        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("声学分片 id 必须唯一")


@dataclass(frozen=True, slots=True)
class PreparedVocals:
    original_id: str
    original_duration: float
    vocals: AudioBuffer
    analysis: VocalAnalysis
    segments: tuple[VocalSegment, ...]
    segment_source: str = "automatic"
    origin: float = 0.0
    separation: dict[str, object] = field(default_factory=dict)
    analysis_description: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.original_id or not isfinite(self.original_duration) or self.original_duration < 0:
            raise ValueError("invalid original identity/duration")
        if self.origin != 0:
            raise ValueError("only declared original-axis origin 0 is supported")
        if self.segment_source not in {"automatic", "manual"}:
            raise ValueError("invalid segment source")
        if abs(self.vocals.duration - self.original_duration) > 1 / self.vocals.sample_rate:
            raise ValueError("stem must cover the complete original duration")
        if self.analysis.source_id != self.vocals.source_id:
            raise ValueError("analysis and vocals have different source identities")
        if self.segment_source == "automatic" and self.segments != self.analysis.segments:
            raise ValueError("automatic selection must preserve analysis segments")
        for segments in (self.analysis.segments, self.segments):
            if len({s.id for s in segments}) != len(segments):
                raise ValueError("duplicate segment ID")
            if any(s.start < 0 or s.end > self.vocals.duration for s in segments):
                raise ValueError("segment outside complete stem")
