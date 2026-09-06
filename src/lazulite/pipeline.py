"""离线对齐组合层。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol
import warnings

from lazulite import __version__
from lazulite.alignment.line import LineAlignmentConfig, align_lines
from lazulite.alignment.offset import OffsetAlignmentConfig, align_with_offset
from lazulite.audio.activity import VocalConfig, analyze_vocals
from lazulite.audio.io import resample_audio
from lazulite.cache import FileCache, cache_key
from lazulite.events import EventSink, ProgressEvent
from lazulite.models.alignment import AlignedLyricLine, AlignmentAnchor, LineAlignmentResult
from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import BackendInfo, TimedText, TranscriptionResult, TranscriptionSegment
from lazulite.transcription.base import Transcriber, TranscriptionRequest
from lazulite.transcription.run import transcribe_segments

_CACHE_ALGORITHM_REVISION = "pipeline-result-v1"


class Separator(Protocol):
    def separate(self, audio: AudioBuffer) -> AudioBuffer: ...


class CacheIdentity(Protocol):
    @property
    def cache_identity(self) -> Mapping[str, object] | None: ...


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
    def __init__(self, *, transcriber: Transcriber, separator: Separator | None = None, cache: FileCache | None = None, event_sink: EventSink | None = None) -> None:
        self._transcriber = transcriber; self._separator = separator; self._cache = cache; self._event_sink = event_sink

    def _emit(self, stage: str, kind: str, message: str | None = None) -> None:
        if self._event_sink: self._event_sink(ProgressEvent(stage, kind, message=message))

    @staticmethod
    def _identity(resource: object) -> Mapping[str, object] | None:
        value = getattr(resource, "cache_identity", None)
        value = value() if callable(value) else value
        return value if isinstance(value, Mapping) else None

    def _cache_key(self, audio: AudioBuffer, document: LyricDocument, config: PipelineConfig) -> str | None:
        if self._cache is None or not audio.source_id.startswith("sha256:"):
            return None
        transcriber = self._identity(self._transcriber)
        separator = self._identity(self._separator) if self._separator is not None else {"kind": "none"}
        # 外部后端没有声明完整有效参数时宁可不缓存，也不跨模型错误复用。
        if transcriber is None or separator is None:
            return None
        return cache_key({
            "stage": "pipeline-result",
            "algorithm_revision": _CACHE_ALGORITHM_REVISION,
            "lazulite_version": __version__,
            "audio": {"source_id": audio.source_id, "sample_rate": audio.sample_rate, "shape": list(audio.samples.shape)},
            "document": asdict(document),
            "pipeline_config": asdict(config),
            "transcriber": transcriber,
            "separator": separator,
        })

    @staticmethod
    def _serialize(result: PipelineResult) -> dict[str, Any]:
        return {"schema": _CACHE_ALGORITHM_REVISION, "result": asdict(result)}

    @staticmethod
    def _deserialize(payload: Mapping[str, Any]) -> PipelineResult | None:
        if payload.get("schema") != _CACHE_ALGORITHM_REVISION or not isinstance(payload.get("result"), Mapping):
            return None
        try:
            result = payload["result"]
            analysis_value = result["analysis"]
            analysis = VocalAnalysis(
                analysis_value["source_id"],
                tuple(VocalSegment(**item) for item in analysis_value["segments"]),
                tuple(analysis_value["diagnostics"]),
            )
            transcription_value = result["transcription"]
            transcription = None
            if transcription_value is not None:
                backend = BackendInfo(**transcription_value["backend"])
                transcription = TranscriptionResult(backend, tuple(
                    TranscriptionSegment(
                        source_segment_id=item["source_segment_id"], text=item["text"], language=item["language"],
                        start=item["start"], end=item["end"], confidence=item["confidence"], confidence_kind=item["confidence_kind"],
                        timed_text=tuple(TimedText(**unit) for unit in item["timed_text"]), backend_metadata=item["backend_metadata"],
                    ) for item in transcription_value["segments"]
                ))
            alignment_value = result["alignment"]
            alignment = LineAlignmentResult(
                tuple(AlignedLyricLine(**item) for item in alignment_value["lines"]),
                alignment_value["strategy"], alignment_value["reliable"],
                tuple(AlignmentAnchor(**item) for item in alignment_value["anchors"]), tuple(alignment_value["diagnostics"]),
            )
            return PipelineResult(analysis, transcription, alignment)
        except (KeyError, TypeError, ValueError) as error:
            warnings.warn(f"缓存结果结构无效，忽略本次命中: {error}", RuntimeWarning)
            return None

    def _cache_result(self, key: str | None, result: PipelineResult) -> PipelineResult:
        if key is not None and self._cache is not None:
            self._cache.set_json(key, self._serialize(result))
        return result

    def _requires_full_distributions(self) -> bool:
        identity = self._identity(self._transcriber)
        config = None if identity is None else identity.get("config")
        return isinstance(config, Mapping) and bool(config.get("distribution_output_dir"))

    @staticmethod
    def _distribution_files_exist(result: PipelineResult) -> bool:
        if result.transcription is None:
            return True
        for segment in result.transcription.segments:
            target = segment.backend_metadata.get("distribution_file")
            if not isinstance(target, str) or not Path(target).is_file():
                return False
        return True

    def run(self, audio: AudioBuffer, document: LyricDocument, *, config: PipelineConfig = PipelineConfig()) -> PipelineResult:
        has_complete_source_times = bool(document.lines) and all(line.source_start is not None for line in document.lines)
        if config.mode == "offset" and not has_complete_source_times:
            raise ValueError("offset 模式要求所有歌词行具有 source_start")
        key = self._cache_key(audio, document, config)
        if key is not None and self._cache is not None:
            cached = self._deserialize(self._cache.get_json(key) or {})
            if cached is not None and (not self._requires_full_distributions() or self._distribution_files_exist(cached)):
                self._emit("pipeline", "completed", "cache_hit")
                return cached
        self._emit("analysis", "started")
        vocals = self._separator.separate(audio) if self._separator else audio
        analysis = analyze_vocals(vocals, config=config.vocal)
        if not analysis.segments:
            # 无人声只是缺少自动化证据，不能丢失调用方传入的歌词行。
            unresolved = LineAlignmentResult(
                tuple(
                    AlignedLyricLine(line.id, None, None, None, None, "unresolved", (), ("no_vocal_evidence",))
                    for line in document.lines
                ),
                "dp",
                False,
                (),
                ("no_vocal_evidence",),
            )
            self._emit("analysis", "completed", "no_vocal_evidence")
            return self._cache_result(key, PipelineResult(analysis, None, unresolved))
        self._emit("transcription", "started")
        transcription = transcribe_segments(resample_audio(vocals, sample_rate=16_000, channels=1), analysis.segments, transcriber=self._transcriber, request=TranscriptionRequest(language=config.language, include_timing=self._transcriber.capabilities.timestamp_granularity != "none"))
        self._emit("transcription", "completed")
        self._emit("alignment", "started")
        if config.mode == "dp" or not has_complete_source_times:
            alignment = align_lines(document, transcription, analysis, config=config.line)
            if config.mode == "hybrid" and not has_complete_source_times:
                alignment = replace(alignment, diagnostics=alignment.diagnostics + ("hybrid_without_source_time",))
        elif config.mode == "offset":
            alignment = align_with_offset(document, transcription, analysis, config=config.offset)
        elif config.mode == "hybrid":
            # 先完成文本 DP，再以独立锚点估计源时间校正；失败保留 DP 结果。
            dp_result = align_lines(document, transcription, analysis, config=config.line)
            refined = align_with_offset(document, transcription, analysis, config=config.offset, guidance=dp_result)
            alignment = replace(refined, strategy="hybrid") if refined.reliable else replace(
                dp_result, strategy="hybrid", diagnostics=dp_result.diagnostics + ("offset_refinement_unreliable",)
            )
        else:
            offset = align_with_offset(document, transcription, analysis, config=config.offset)
            if offset.reliable:
                alignment = offset
            else:
                dp_result = align_lines(document, transcription, analysis, config=config.line)
                refined = align_with_offset(document, transcription, analysis, config=config.offset, guidance=dp_result)
                alignment = replace(refined, strategy="hybrid", diagnostics=("offset_fallback",)) if refined.reliable else replace(
                    dp_result, diagnostics=dp_result.diagnostics + ("offset_fallback", "offset_refinement_unreliable")
                )
        self._emit("alignment", "completed")
        return self._cache_result(key, PipelineResult(analysis, transcription, alignment))
