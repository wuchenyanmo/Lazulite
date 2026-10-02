"""给定声学窗口的转写调度，不加载或选择具体模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal
from .evidence import quality, repetition, eligible_for_transcription

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.transcription import TranscriptionResult

from .base import Transcriber, TranscriptionInput, TranscriptionRequest


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    prompt_mode: Literal["none", "previous", "hint", "hybrid"] = "previous"
    lyric_hint: str | None = None
    max_prompt_chars: int = 180
    max_previous_chunks: int = 2
    max_hint_chars: int = 80

    def __post_init__(self):
        if self.prompt_mode not in {"none", "previous", "hint", "hybrid"}:
            raise ValueError("unknown prompt mode")
        if (
            min(self.max_prompt_chars, self.max_previous_chunks, self.max_hint_chars)
            < 1
        ):
            raise ValueError("context limits must be positive")


def _context(history, segment, config, explicit):
    if segment.off_center_risk is not None and segment.off_center_risk >= 0.55:
        return None
    parts = [explicit] if explicit else []
    if config.prompt_mode in {"previous", "hybrid"}:
        parts += history[-config.max_previous_chunks :]
    if config.prompt_mode in {"hint", "hybrid"} and (
        segment.off_center_risk is None or segment.off_center_risk < 0.40
    ):
        hint = " / ".join(dict.fromkeys((config.lyric_hint or "").splitlines()))
        if hint:
            parts.append(hint[: config.max_hint_chars])
    return " ".join(parts)[-config.max_prompt_chars :] or None


def _split_segment(
    segment: VocalSegment, maximum_seconds: float
) -> tuple[VocalSegment, ...]:
    """将超长窗口显式拆开，同时保留原始声学分片 ID 作为证据关联键。"""
    if segment.end - segment.start <= maximum_seconds:
        return (segment,)
    windows: list[VocalSegment] = []
    start = segment.start
    while start < segment.end:
        end = min(start + maximum_seconds, segment.end)
        core_start = max(start, min(segment.core_start, end))
        core_end = min(end, max(segment.core_end, start))
        # 当核心区间落在其他窗口时，窗口本身仍是有效的位置证据。
        if core_end <= core_start:
            core_start, core_end = start, end
        windows.append(
            VocalSegment(
                id=segment.id,
                section_id=segment.section_id,
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                presence=segment.presence,
                off_center_risk=segment.off_center_risk,
            )
        )
        start = end
    return tuple(windows)


def build_transcription_inputs(
    audio: AudioBuffer,
    segments: Sequence[VocalSegment],
    *,
    maximum_seconds: float | None,
) -> tuple[TranscriptionInput, ...]:
    """构造并按后端上限拆分输入；不会截断后把残缺窗口当作完整成功。"""
    expanded: list[TranscriptionInput] = []
    for segment in segments:
        windows = (
            _split_segment(segment, maximum_seconds) if maximum_seconds else (segment,)
        )
        expanded.extend(TranscriptionInput(audio, window) for window in windows)
    return tuple(expanded)


def transcribe_segments(
    audio: AudioBuffer,
    segments: Sequence[VocalSegment],
    *,
    transcriber: Transcriber,
    request: TranscriptionRequest | None = None,
    config: TranscriptionConfig = TranscriptionConfig(),
) -> TranscriptionResult:
    """调用已注入的后端，并在边界检查它返回的证据关联。"""
    effective_request = request or TranscriptionRequest()
    if effective_request.context and not transcriber.capabilities.context:
        raise ValueError(f"后端 {transcriber.info.name} 不支持上下文转写")
    if effective_request.context and effective_request.include_timing:
        raise ValueError("context and fine timing cannot be requested together")
    if (
        config.prompt_mode in {"hint", "hybrid"}
        and config.lyric_hint
        and not transcriber.capabilities.context
    ):
        raise ValueError("backend does not support lyric hints")
    inputs = build_transcription_inputs(
        audio,
        tuple(s for s in segments if eligible_for_transcription(s)),
        maximum_seconds=transcriber.capabilities.max_audio_seconds,
    )
    if not inputs:
        return TranscriptionResult(transcriber.info, ())
    contextual = (
        transcriber.capabilities.context and not effective_request.include_timing
    )
    if contextual:
        history = []
        output = []
        for item in inputs:
            prompt = _context(history, item.segment, config, effective_request.context)
            result = transcriber.transcribe(
                (item,), request=replace(effective_request, context=prompt)
            )
            for segment in result:
                if segment.source_segment_id != item.segment.id:
                    raise ValueError("backend returned an unrelated context result")
                output.append(
                    replace(
                        segment,
                        backend_metadata={
                            **segment.backend_metadata,
                            "request_context": prompt,
                            "request_timing": effective_request.include_timing,
                            "request_window": [item.segment.start, item.segment.end],
                        },
                    )
                )
                if quality(segment.text, item.segment).risk < 0.42:
                    history.append(repetition(segment.text)[1])
    else:
        output = transcriber.transcribe(
            inputs, request=replace(effective_request, context=None)
        )
    valid_ids = {segment.id for segment in segments}
    unknown_ids = {segment.source_segment_id for segment in output} - valid_ids
    if unknown_ids:
        raise ValueError(f"后端返回了未知 source_segment_id: {sorted(unknown_ids)!r}")
    output = tuple(
        replace(
            item,
            backend_metadata={
                **item.backend_metadata,
                "request_timing": effective_request.include_timing,
                "request_windows": [
                    [value.segment.start, value.segment.end]
                    for value in inputs
                    if value.segment.id == item.source_segment_id
                ],
            },
        )
        for item in output
    )
    return TranscriptionResult(backend=transcriber.info, segments=output)
