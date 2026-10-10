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
from lazulite.events import EventSink, ProgressEvent
from lazulite.models.alignment import (
    AlignedLyricLine,
    AlignmentAnchor,
    LineAlignmentResult,
)
from lazulite.models.audio import AudioBuffer, PreparedVocals, VocalAnalysis, VocalSegment
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
from lazulite.transcription.run import TranscriptionConfig, TranscriptionStage, run_transcription_stage, build_transcription_inputs
from lazulite.transcription.evidence import eligible_for_transcription

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
    transcriptions: dict[str, TranscriptionStage] = field(default_factory=dict)


class AlignmentPipeline:
    def __init__(
        self,
        *,
        transcriber: Transcriber | None = None,
        separator: object | None = None,
        event_sink: EventSink | None = None,
        cancel=None,
    ) -> None:
        self._transcriber = transcriber
        self._separator = separator
        self._event_sink = event_sink
        self._cancel = cancel

    def _emit(self, stage: str, kind: str, message: str | None = None) -> None:
        from lazulite.events import checkpoint
        checkpoint(self._cancel)
        if self._event_sink:
            self._event_sink(ProgressEvent(stage, kind, message=message))

    def run(
        self,
        audio: AudioBuffer,
        document: LyricDocument,
        *,
        config: PipelineConfig = PipelineConfig(),
        transcriptions: dict[str, TranscriptionStage] | None = None,
    ) -> PipelineResult:
        prepared = (
            prepare_vocals(audio, separator=self._separator, config=config.vocal, cancel=self._cancel, event_sink=self._event_sink)
            if self._separator is not None else
            prepare_existing_vocals(audio, original_id=audio.source_id,
                                    original_duration=audio.duration, config=config.vocal)
        )
        return self.run_prepared(prepared, document, config=config, transcriptions=transcriptions)

    def run_prepared(
        self, prepared: PreparedVocals, document: LyricDocument, *,
        config: PipelineConfig = PipelineConfig(),
        transcriptions: dict[str, TranscriptionStage] | None = None,
    ) -> PipelineResult:
        from lazulite.events import checkpoint
        checkpoint(self._cancel)
        prepared.__post_init__()
        audio = vocals = prepared.vocals
        analysis = replace(prepared.analysis, segments=prepared.segments)
        has_complete_source_times = bool(document.lines) and all(
            line.source_start is not None for line in document.lines
        )
        if config.mode == "offset" and not has_complete_source_times:
            raise ValueError("offset 模式要求所有歌词行具有 source_start")
        execution = {
            "algorithm": "line-request-anchors",
            "config": asdict(config),
            "audio_input": {
                "sample_rate": audio.sample_rate,
                "channels": audio.samples.shape[0],
            },
            "vocal_reference": prepared.analysis_description.get("reference"),
            "asr_input": "mid_minus_absolute_side",
            "resampling": "ffmpeg-swr:filter_size=32:phase_shift=10:cutoff=0.97:zero_pad=64-output-samples:length=ceil",
            "requests": [],
        }
        self._emit("analysis", "started")
        if (prepared.segment_source == "automatic" and analysis.is_vocal is False) or not analysis.segments:
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
            return PipelineResult(analysis, None, unresolved, execution)
        self._emit("analysis", "completed")
        self._emit("transcription", "started")
        asr_audio = resample_audio(
            main_vocal_audio(vocals), sample_rate=16_000, channels=1
        )
        supplied = {} if transcriptions is None else dict(transcriptions)
        if set(supplied) - {"preselected", "full"}:
            raise ValueError("unknown transcription stage")
        for stage in supplied.values():
            stage.validate(asr_audio)
        capabilities = self._transcriber.capabilities if self._transcriber is not None else next(
            (stage.capabilities for stage in supplied.values()), None)
        if capabilities is None:
            raise ValueError("provide transcription stages or a transcriber")
        timing = capabilities.timestamp_granularity != "none"
        stages = {}
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

        def transcribe(name, segments, include_timing):
            execution["requests"].append(
                {
                    "segments": [asdict(s) for s in segments],
                    "include_timing": include_timing,
                    "context_config": asdict(run_config),
                }
            )
            request = TranscriptionRequest(language=config.language, include_timing=include_timing)
            if name in supplied:
                stage = supplied[name]
                if (stage.segments != tuple(segments) or stage.request != request or
                    stage.config != run_config or stage.capabilities != capabilities):
                    raise ValueError(f"transcription stage {name} has different windows/context/timing conditions")
            else:
                if self._transcriber is None:
                    raise ValueError(f"missing transcription stage: {name}")
                prior = stages.get("preselected") if name == "full" and not include_timing and (
                    not capabilities.context or run_config.prompt_mode == "none") else None
                remaining = tuple(s for s in segments if prior is None or s.id not in {p.id for p in prior.segments})
                stage = run_transcription_stage(asr_audio, remaining, transcriber=self._transcriber,
                    request=request, config=run_config, cancel=self._cancel, event_sink=self._event_sink)
                if prior is not None:
                    rows={s.request_window.request_id:s for s in prior.result.segments+stage.result.segments}
                    inputs=build_transcription_inputs(asr_audio,segments,maximum_seconds=capabilities.max_audio_seconds)
                    result=TranscriptionResult(stage.result.backend,tuple(
                        replace(rows[i.segment.request_id],request_window=i.segment) for i in inputs))
                    stage=TranscriptionStage(tuple(segments),request,run_config,capabilities,result)
                    stage.validate(asr_audio)
            stages[name] = stage
            return stage.result

        preselected = config.mode in {"offset", "auto"} and has_complete_source_times
        if preselected:
            selected = select_offset_segments(all_segments)
            transcription = transcribe("preselected", selected, False)
            offset = align_with_offset(
                document, transcription, analysis, config=config.offset
            )
            if config.mode == "offset" or offset.offset_accepted:
                self._emit("transcription", "completed")
                self._emit("alignment", "completed")
                return PipelineResult(analysis, transcription, offset, execution, stages)
            transcription = transcribe("full", all_segments, timing)
        else:
            transcription = transcribe("full", all_segments, timing)
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
        return PipelineResult(analysis, transcription, alignment, execution, stages)


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
