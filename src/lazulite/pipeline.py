"""离线对齐组合层。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

from lazulite.alignment.line import LineAlignmentConfig, align_lines
from lazulite.alignment.offset import OffsetAlignmentConfig, align_with_offset
from lazulite.audio.activity import VocalConfig, analyze_vocals
from lazulite.audio.io import resample_audio
from lazulite.events import EventSink, ProgressEvent
from lazulite.models.alignment import LineAlignmentResult
from lazulite.models.audio import AudioBuffer, VocalAnalysis
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult
from lazulite.transcription.base import Transcriber, TranscriptionRequest
from lazulite.transcription.run import transcribe_segments


class Separator(Protocol):
    def separate(self, audio: AudioBuffer) -> AudioBuffer: ...


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    mode: Literal["dp", "offset", "hybrid", "auto"] = "auto"
    language: str | None = None
    vocal: VocalConfig = VocalConfig()
    line: LineAlignmentConfig = LineAlignmentConfig()
    offset: OffsetAlignmentConfig = OffsetAlignmentConfig()


@dataclass(frozen=True, slots=True)
class PipelineResult:
    analysis: VocalAnalysis
    transcription: TranscriptionResult | None
    alignment: LineAlignmentResult


class AlignmentPipeline:
    def __init__(self, *, transcriber: Transcriber, separator: Separator | None = None, event_sink: EventSink | None = None) -> None:
        self._transcriber = transcriber; self._separator = separator; self._event_sink = event_sink

    def _emit(self, stage: str, kind: str, message: str | None = None) -> None:
        if self._event_sink: self._event_sink(ProgressEvent(stage, kind, message=message))

    def run(self, audio: AudioBuffer, document: LyricDocument, *, config: PipelineConfig = PipelineConfig()) -> PipelineResult:
        self._emit("analysis", "started")
        vocals = self._separator.separate(audio) if self._separator else audio
        analysis = analyze_vocals(vocals, config=config.vocal)
        if not analysis.segments:
            unresolved = LineAlignmentResult(tuple(), "dp", False, (), ("no_vocal_evidence",))
            self._emit("analysis", "completed", "no_vocal_evidence")
            return PipelineResult(analysis, None, unresolved)
        self._emit("transcription", "started")
        transcription = transcribe_segments(resample_audio(vocals, sample_rate=16_000, channels=1), analysis.segments, transcriber=self._transcriber, request=TranscriptionRequest(language=config.language, include_timing=self._transcriber.capabilities.timestamp_granularity != "none"))
        has_source_times = any(line.source_start is not None for line in document.lines)
        if config.mode == "dp" or (config.mode == "auto" and not has_source_times): alignment = align_lines(document, transcription, analysis, config=config.line)
        else:
            offset = align_with_offset(document, transcription, analysis, config=config.offset)
            alignment = offset if offset.reliable else align_lines(document, transcription, analysis, config=config.line)
            if config.mode == "hybrid" and alignment.strategy == "dp": alignment = replace(alignment, strategy="hybrid")
        self._emit("alignment", "completed")
        return PipelineResult(analysis, transcription, alignment)
