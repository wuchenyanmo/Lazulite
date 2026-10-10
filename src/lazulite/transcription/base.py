"""ASR 后端的最小公共协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.transcription import BackendInfo, TranscriptionSegment, RequestWindow

TimestampGranularity = Literal["none", "segment", "subword", "word", "character"]


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    language: str | None = None
    include_timing: bool = False
    context: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriberCapabilities:
    timestamp_granularity: TimestampGranularity
    context: bool
    max_audio_seconds: float | None

    def __post_init__(self) -> None:
        if self.max_audio_seconds is not None and self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds 必须为正数或 None")


@dataclass(frozen=True, slots=True)
class TranscriptionInput:
    audio: AudioBuffer
    segment: VocalSegment | RequestWindow

    def __post_init__(self) -> None:
        if self.segment.end > self.audio.duration:
            raise ValueError("转写分片不能超出音频时长")


class Transcriber(Protocol):
    """模型实例的协议；调用方可以注入任意符合此协议的外部实现。"""

    info: BackendInfo
    capabilities: TranscriberCapabilities

    def transcribe(
        self,
        inputs: Sequence[TranscriptionInput],
        *,
        request: TranscriptionRequest,
    ) -> tuple[TranscriptionSegment, ...]: ...

    def close(self) -> None: ...
