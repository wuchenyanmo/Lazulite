"""给定声学窗口的转写调度，不加载或选择具体模型。"""

from __future__ import annotations

from collections.abc import Sequence

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.transcription import TranscriptionResult

from .base import Transcriber, TranscriptionInput, TranscriptionRequest


def _split_segment(segment: VocalSegment, maximum_seconds: float) -> tuple[VocalSegment, ...]:
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
        windows = _split_segment(segment, maximum_seconds) if maximum_seconds else (segment,)
        expanded.extend(TranscriptionInput(audio, window) for window in windows)
    return tuple(expanded)


def transcribe_segments(
    audio: AudioBuffer,
    segments: Sequence[VocalSegment],
    *,
    transcriber: Transcriber,
    request: TranscriptionRequest | None = None,
) -> TranscriptionResult:
    """调用已注入的后端，并在边界检查它返回的证据关联。"""
    effective_request = request or TranscriptionRequest()
    if effective_request.context and not transcriber.capabilities.context:
        raise ValueError(f"后端 {transcriber.info.name} 不支持上下文转写")
    inputs = build_transcription_inputs(
        audio, segments, maximum_seconds=transcriber.capabilities.max_audio_seconds
    )
    output = transcriber.transcribe(inputs, request=effective_request)
    valid_ids = {segment.id for segment in segments}
    unknown_ids = {segment.source_segment_id for segment in output} - valid_ids
    if unknown_ids:
        raise ValueError(f"后端返回了未知 source_segment_id: {sorted(unknown_ids)!r}")
    return TranscriptionResult(backend=transcriber.info, segments=tuple(output))
