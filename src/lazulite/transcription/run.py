"""给定声学窗口的转写调度，不加载或选择具体模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal
from .evidence import transcription_quality, EvidenceConfig, repetition, eligible_for_transcription

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.transcription import TranscriptionResult, RequestWindow

from .base import Transcriber, TranscriptionInput, TranscriptionRequest, TranscriberCapabilities


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    evidence: EvidenceConfig = EvidenceConfig()
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


def build_transcription_inputs(audio, segments, *, maximum_seconds):
    """Every actual request has a distinct identity and optional true core intersection."""
    expanded=[]
    if maximum_seconds is not None and maximum_seconds <= 0:
        raise ValueError('maximum_seconds must be positive')
    for segment in segments:
        start=segment.start
        while start < segment.end:
            end=min(segment.end,start+maximum_seconds) if maximum_seconds else segment.end
            # Request bounds represent the actual PCM sample slice, not unrounded seconds.
            start_sample=round(start*audio.sample_rate); end_sample=round(end*audio.sample_rate)
            a,b=start_sample/audio.sample_rate,end_sample/audio.sample_rate
            core_start,core_end=max(a,segment.core_start),min(b,segment.core_end)
            if core_end <= core_start:
                core_start=core_end=None
            window=RequestWindow(f'{segment.id}:{start_sample}:{end_sample}',
                segment.id,a,b,core_start,core_end,len(expanded),segment.section_id,segment.presence,segment.off_center_risk)
            expanded.append(TranscriptionInput(audio,window))
            start=end
    return tuple(expanded)


def _bind(item, result, request, context=None):
    if result.source_segment_id != item.segment.id:
        raise ValueError('backend returned unrelated request evidence')
    return replace(result,request_window=item.segment,backend_metadata={**result.backend_metadata,
        'request_context':context,'request_timing':request.include_timing,'request_language':request.language})


def transcribe_segments(
    audio: AudioBuffer,
    segments: Sequence[VocalSegment],
    *,
    transcriber: Transcriber,
    request: TranscriptionRequest | None = None,
    config: TranscriptionConfig = TranscriptionConfig(),
    cancel=None, event_sink=None,
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
    from lazulite.events import checkpoint
    completed=0
    def generate(batch, request):
        nonlocal completed
        checkpoint(cancel,event_sink,'transcription',completed,len(inputs))
        values=transcriber.transcribe(batch,request=request)
        completed+=len(batch)
        checkpoint(cancel,event_sink,'transcription',completed,len(inputs))
        return values
    contextual = (
        transcriber.capabilities.context and not effective_request.include_timing
    )
    if contextual:
        history = []
        output = []
        for item in inputs:
            prompt = _context(history, item.segment, config, effective_request.context)
            result = generate(
                (item,), request=replace(effective_request, context=prompt)
            )
            if len(result)!=1:
                raise ValueError('backend must return one result per request window')
            segment=_bind(item,result[0],effective_request,prompt)
            output.append(segment)
            if transcription_quality(segment,item.segment,config.evidence).risk < .42:
                history.append(repetition(segment.text)[1])
    else:
        output = generate(
            inputs, request=replace(effective_request, context=None)
        )
        if len(output)!=len(inputs):
            raise ValueError('backend must return one ordered result per request window')
        output=tuple(_bind(item,result,effective_request) for item,result in zip(inputs,output))
    from .timing import organize_timing
    return organize_timing(TranscriptionResult(backend=transcriber.info,segments=tuple(output)))


@dataclass(frozen=True, slots=True)
class TranscriptionStage:
    """A complete set of windows under one timing/context generation condition."""
    segments: tuple[VocalSegment, ...]
    request: TranscriptionRequest
    config: TranscriptionConfig
    capabilities: TranscriberCapabilities
    result: TranscriptionResult

    def validate(self, audio):
        expected = build_transcription_inputs(audio, self.segments,
            maximum_seconds=self.capabilities.max_audio_seconds)
        if len(expected) != len(self.result.segments):
            raise ValueError("transcription stage missing request windows")
        history=[]
        for item, result in zip(expected, self.result.segments):
            result.__post_init__()
            if result.request_window != item.segment or result.source_segment_id != item.segment.id:
                raise ValueError("transcription window/parent mismatch")
            if result.backend_metadata.get('request_timing') is not self.request.include_timing:
                raise ValueError("transcription timing condition mismatch")
            contextual = self.capabilities.context and not self.request.include_timing
            context = _context(history,item.segment,self.config,self.request.context) if contextual else None
            if ('request_context' not in result.backend_metadata or result.backend_metadata['request_context']!=context or
                'request_language' not in result.backend_metadata or result.backend_metadata['request_language']!=self.request.language):
                raise ValueError("transcription actual context/language condition mismatch")
            if contextual and transcription_quality(result,item.segment,self.config.evidence).risk < .42:
                history.append(repetition(result.text)[1])


def run_transcription_stage(audio, segments, *, transcriber, request=TranscriptionRequest(),
                            config=TranscriptionConfig(), cancel=None, event_sink=None):
    selected = tuple(s for s in segments if eligible_for_transcription(s))
    result = transcribe_segments(audio, selected, transcriber=transcriber, request=request,
                                config=config, cancel=cancel, event_sink=event_sink)
    stage = TranscriptionStage(selected, request, config, transcriber.capabilities, result)
    stage.validate(audio)
    return stage
