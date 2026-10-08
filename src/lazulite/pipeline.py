"""离线对齐组合层。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol
import warnings

from lazulite import __version__
from lazulite.alignment.line import LineAlignmentConfig, align_lines
from lazulite.alignment.offset import OffsetAlignmentConfig, align_with_offset
from lazulite.audio.activity import VocalConfig, main_vocal_audio
from lazulite.audio.io import resample_audio
from lazulite.audio.preparation import prepare_vocals, prepare_existing_vocals
from lazulite.cache import FileCache, cache_key
from lazulite.events import EventSink, ProgressEvent
from lazulite.models.alignment import (
    AlignedLyricLine,
    AlignmentAnchor,
    LineAlignmentResult,
)
from lazulite.models.audio import AudioBuffer, PreparedVocals, VocalAnalysis, VocalSegment, pcm_identity
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import (
    BackendInfo,
    TimedText,
    RequestWindow,
    ProbabilitySummary,
    TranscriptionResult,
    TranscriptionSegment,
)
from lazulite.transcription.base import Transcriber, TranscriptionRequest
from lazulite.transcription.run import TranscriptionConfig, transcribe_segments
from lazulite.transcription.evidence import eligible_for_transcription

_CACHE_ALGORITHM_REVISION = "pipeline-result-v5-coverage"


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
    transcription: TranscriptionConfig = TranscriptionConfig()


@dataclass(frozen=True, slots=True)
class PipelineResult:
    analysis: VocalAnalysis
    transcription: TranscriptionResult | None
    alignment: LineAlignmentResult
    execution: dict[str, object] = field(default_factory=dict)


class AlignmentPipeline:
    def __init__(
        self,
        *,
        transcriber: Transcriber,
        separator: Separator | None = None,
        cache: FileCache | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._separator = separator
        self._cache = cache
        self._event_sink = event_sink

    def _emit(self, stage: str, kind: str, message: str | None = None) -> None:
        if self._event_sink:
            self._event_sink(ProgressEvent(stage, kind, message=message))

    @staticmethod
    def _identity(resource: object) -> Mapping[str, object] | None:
        value = getattr(resource, "cache_identity", None)
        value = value() if callable(value) else value
        return value if isinstance(value, Mapping) else None

    def _cache_key(
        self, audio: AudioBuffer, document: LyricDocument, config: PipelineConfig
    ) -> str | None:
        if self._cache is None or not audio.source_id.startswith("sha256:"):
            return None
        transcriber = self._identity(self._transcriber)
        separator = (
            self._identity(self._separator)
            if self._separator is not None
            else {"kind": "none"}
        )
        # 外部后端没有声明完整有效参数时宁可不缓存，也不跨模型错误复用。
        if transcriber is None or separator is None:
            return None
        return cache_key(
            {
                "stage": "pipeline-result",
                "algorithm_revision": _CACHE_ALGORITHM_REVISION,
                "lazulite_version": __version__,
                "audio": {
                    "source_id": audio.source_id,
                    "pcm_id": pcm_identity(audio),
                    "sample_rate": audio.sample_rate,
                    "shape": list(audio.samples.shape),
                },
                "document": asdict(document),
                "pipeline_config": asdict(config),
                "transcriber": transcriber,
                "separator": separator,
            }
        )

    @staticmethod
    def _serialize(result: PipelineResult) -> dict[str, Any]:
        return {"schema": _CACHE_ALGORITHM_REVISION, "result": asdict(result)}

    @staticmethod
    def _deserialize(payload: Mapping[str, Any]) -> PipelineResult | None:
        if payload.get("schema") != _CACHE_ALGORITHM_REVISION or not isinstance(
            payload.get("result"), Mapping
        ):
            return None
        try:
            result = payload["result"]
            analysis_value = result["analysis"]
            analysis = VocalAnalysis(
                analysis_value["source_id"],
                tuple(VocalSegment(**item) for item in analysis_value["segments"]),
                tuple(analysis_value["diagnostics"]),
                analysis_value["is_vocal"],
                analysis_value["vocal_seconds"],
            )
            transcription_value = result["transcription"]
            transcription = None
            if transcription_value is not None:
                backend = BackendInfo(**transcription_value["backend"])
                transcription = TranscriptionResult(
                    backend,
                    tuple(
                        TranscriptionSegment(
                            source_segment_id=item["source_segment_id"],
                            text=item["text"],
                            language=item["language"],
                            start=item["start"],
                            end=item["end"],
                            confidence=item["confidence"],
                            confidence_kind=item["confidence_kind"],
                            timed_text=tuple(
                                TimedText(**{**unit, "generation_steps":tuple(unit["generation_steps"]), "probability":ProbabilitySummary(**unit["probability"]) if unit["probability"] is not None else None}) for unit in item["timed_text"]
                            ),
                            backend_metadata=item["backend_metadata"],
                            request_window=RequestWindow(**item["request_window"]) if item["request_window"] is not None else None,
                            probability=ProbabilitySummary(**item["probability"]) if item["probability"] is not None else None,
                            issues=tuple(item["issues"]),
                        )
                        for item in transcription_value["segments"]
                    ),
                )
            alignment_value = result["alignment"]
            alignment = LineAlignmentResult(
                tuple(
                    AlignedLyricLine(
                        **{
                            **item,
                            "source_segment_ids": tuple(item["source_segment_ids"]),
                            "issues": tuple(item["issues"]),
                            "source_request_ids":tuple(item["source_request_ids"]),
                        }
                    )
                    for item in alignment_value["lines"]
                ),
                alignment_value["strategy"],
                alignment_value["offset_accepted"],
                tuple(AlignmentAnchor(**item) for item in alignment_value["anchors"]),
                tuple(alignment_value["diagnostics"]),
            )
            return PipelineResult(
                analysis, transcription, alignment, dict(result.get("execution", {}))
            )
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
        return isinstance(config, Mapping) and bool(
            config.get("distribution_output_dir")
        )

    @staticmethod
    def _distribution_files_exist(result: PipelineResult) -> bool:
        if result.transcription is None:
            return True
        for segment in result.transcription.segments:
            target = segment.backend_metadata.get("distribution_file")
            if not isinstance(target, str) or not Path(target).is_file():
                return False
        return True

    def run(
        self,
        audio: AudioBuffer,
        document: LyricDocument,
        *,
        config: PipelineConfig = PipelineConfig(),
    ) -> PipelineResult:
        prepared = (
            prepare_vocals(audio, separator=self._separator, config=config.vocal)
            if self._separator is not None else
            prepare_existing_vocals(audio, original_id=audio.source_id,
                                    original_duration=audio.duration, config=config.vocal)
        )
        return self.run_prepared(prepared, document, config=config)

    def run_prepared(
        self, prepared: PreparedVocals, document: LyricDocument, *,
        config: PipelineConfig = PipelineConfig(),
    ) -> PipelineResult:
        prepared.__post_init__()
        audio = vocals = prepared.vocals
        analysis = replace(prepared.analysis, segments=prepared.segments)
        has_complete_source_times = bool(document.lines) and all(
            line.source_start is not None for line in document.lines
        )
        if config.mode == "offset" and not has_complete_source_times:
            raise ValueError("offset 模式要求所有歌词行具有 source_start")
        key = self._cache_key(audio, document, config)
        if key is not None:
            key = cache_key({"pipeline": key, "pcm": prepared.pcm_id,
                             "analysis": asdict(prepared.analysis),
                             "selection": [asdict(s) for s in prepared.segments],
                             "source": prepared.segment_source,
                             "preparation": prepared.analysis_identity,
                             "separation": prepared.separation})
        if key is not None and self._cache is not None:
            cached = self._deserialize(self._cache.get_json(key) or {})
            if cached is not None and (
                not self._requires_full_distributions()
                or self._distribution_files_exist(cached)
            ):
                self._emit("pipeline", "completed", "cache_hit")
                return cached
        execution = {
            "algorithm_revision": _CACHE_ALGORITHM_REVISION,
            "config": asdict(config),
            "audio_input": {
                "sample_rate": audio.sample_rate,
                "channels": audio.samples.shape[0],
            },
            "vocal_reference": prepared.analysis_identity.get("reference"),
            "prepared_pcm": prepared.pcm_id,
            "asr_input": "mid_minus_absolute_side",
            "resampling": "ffmpeg-swr:filter_size=32:phase_shift=10:cutoff=0.97:zero_pad=64-output-samples:length=ceil",
            "requests": [],
        }
        self._emit("analysis", "started")
        if not analysis.is_vocal or not analysis.segments:
            # 无人声只是缺少自动化证据，不能丢失调用方传入的歌词行。
            unresolved = LineAlignmentResult(
                tuple(
                    AlignedLyricLine(
                        line.id,
                        None,
                        None,
                        None,
                        None,
                        "unresolved",
                        (),
                        ("no_vocal_evidence",),
                    )
                    for line in document.lines
                ),
                "dp",
                False,
                (),
                ("no_vocal_evidence",),
            )
            self._emit("analysis", "completed", "no_vocal_evidence")
            return self._cache_result(
                key, PipelineResult(analysis, None, unresolved, execution)
            )
        self._emit("analysis", "completed")
        self._emit("transcription", "started")
        asr_audio = resample_audio(
            main_vocal_audio(vocals), sample_rate=16_000, channels=1
        )
        timing = self._transcriber.capabilities.timestamp_granularity != "none"
        all_segments = tuple(
            s for s in analysis.segments if eligible_for_transcription(s)
        )
        run_config = config.transcription
        if (
            run_config.prompt_mode in {"hint", "hybrid"}
            and run_config.lyric_hint is None
        ):
            run_config = replace(
                run_config, lyric_hint="\n".join(line.text for line in document.lines)
            )

        def transcribe(segments, include_timing):
            execution["requests"].append(
                {
                    "segments": [asdict(s) for s in segments],
                    "include_timing": include_timing,
                    "context_config": asdict(run_config),
                }
            )
            return transcribe_segments(
                asr_audio,
                segments,
                transcriber=self._transcriber,
                request=TranscriptionRequest(
                    language=config.language, include_timing=include_timing
                ),
                config=run_config,
            )

        preselected = config.mode in {"offset", "auto"} and has_complete_source_times
        if preselected:
            selected = select_offset_segments(all_segments)
            transcription = transcribe(selected, False)
            offset = align_with_offset(
                document, transcription, analysis, config=config.offset
            )
            if config.mode == "offset" or offset.offset_accepted:
                self._emit("transcription", "completed")
                self._emit("alignment", "completed")
                return self._cache_result(
                    key, PipelineResult(analysis, transcription, offset, execution)
                )
            # Timing/context changes require new generation. Stateless requests reuse exact windows.
            if timing or (
                self._transcriber.capabilities.context
                and run_config.prompt_mode != "none"
            ):
                transcription = transcribe(all_segments, timing)
            else:
                reused = {s.id for s in selected}
                additional = transcribe(
                    tuple(s for s in all_segments if s.id not in reused), False
                )
                order = {s.id: i for i, s in enumerate(all_segments)}
                combined = transcription.segments + additional.segments
                transcription = TranscriptionResult(
                    transcription.backend,
                    tuple(sorted(combined, key=lambda s: order[s.source_segment_id])),
                )
        else:
            transcription = transcribe(all_segments, timing)
        self._emit("transcription", "completed")
        self._emit("alignment", "started")
        alignment = align_lines(document, transcription, analysis, config=config.line)
        if config.mode == "hybrid" and not has_complete_source_times:
            alignment = replace(
                alignment,
                diagnostics=alignment.diagnostics + ("hybrid_without_source_time",),
            )
        if has_complete_source_times and config.mode in {"hybrid", "auto"}:
            refined = align_with_offset(
                document,
                transcription,
                analysis,
                config=config.offset,
                guidance=alignment,
            )
            alignment = (
                replace(refined, strategy="hybrid")
                if refined.offset_accepted
                else replace(
                    alignment,
                    diagnostics=alignment.diagnostics
                    + ("offset_refinement_unreliable",),
                )
            )
        if preselected:
            alignment = replace(
                alignment, diagnostics=alignment.diagnostics + ("offset_fallback",)
            )
        self._emit("alignment", "completed")
        return self._cache_result(
            key, PipelineResult(analysis, transcription, alignment, execution)
        )


def select_offset_segments(segments):
    by_section = {}
    for segment in segments:
        by_section.setdefault(segment.section_id, []).append(segment)
    selected = set()

    def priority(segment):
        return (
            1.10 * (segment.presence if segment.presence is not None else 0)
            - 0.55
            * (segment.off_center_risk if segment.off_center_risk is not None else 0)
            + 0.05 * min(segment.core_end - segment.core_start, 8)
        )

    for group in by_section.values():
        candidates = [
            s
            for s in group
            if (s.presence is None or s.presence >= 0.48)
            and (s.off_center_risk is None or s.off_center_risk <= 0.50)
            and s.end - s.start >= 1
        ]
        selected.update(
            s.id for s in sorted(candidates, key=lambda s: (-priority(s), s.start))[:3]
        )
    if not selected:
        selected.update(
            s.id
            for s in segments
            if (s.presence is None or s.presence >= 0.52)
            and (s.off_center_risk is None or s.off_center_risk <= 0.45)
        )
    return tuple(s for s in segments if s.id in selected)
