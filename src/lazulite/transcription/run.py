"""给定声学窗口的转写调度，不加载或选择具体模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal
from .evidence import transcription_quality, EvidenceConfig, repetition, eligible_for_transcription

from lazulite.models.audio import AudioBuffer, VocalSegment, pcm_identity
from lazulite.models.transcription import TranscriptionResult, RequestWindow, transcription_segment_from_dict
from lazulite.cache import cache_key

from .base import Transcriber, TranscriptionInput, TranscriptionRequest


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
            window=RequestWindow(cache_key({'source':audio.source_id,'parent':segment.id,'samples':[start_sample,end_sample],'rate':audio.sample_rate}),
                segment.id,a,b,core_start,core_end,len(expanded),segment.section_id,segment.presence,segment.off_center_risk)
            expanded.append(TranscriptionInput(audio,window))
            start=end
    return tuple(expanded)


def _bind(item, result, request, context=None):
    if result.source_segment_id != item.segment.id:
        raise ValueError('backend returned unrelated request evidence')
    return replace(result,request_window=item.segment,backend_metadata={**result.backend_metadata,
        'request_context':context,'request_timing':request.include_timing})


def transcribe_segments(
    audio: AudioBuffer,
    segments: Sequence[VocalSegment],
    *,
    transcriber: Transcriber,
    request: TranscriptionRequest | None = None,
    config: TranscriptionConfig = TranscriptionConfig(),
    cache=None, cancel=None, event_sink=None,
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
    identity=getattr(transcriber,'cache_identity',None)
    identity=identity() if callable(identity) else identity
    pcm=pcm_identity(audio) if cache is not None and identity is not None else None
    completed=0
    def generate(batch, request):
        nonlocal completed
        # Preserve batch behavior without cache; with cache, each exact request is independent.
        checkpoint(cancel,event_sink,'transcription',completed,len(inputs))
        if cache is None or identity is None:
            values=transcriber.transcribe(batch,request=request)
            completed+=len(batch)
            checkpoint(cancel,event_sink,'transcription',completed,len(inputs))
            return values
        values=[]
        for item in batch:
            key=cache_key({'schema':'transcription-request-v1','pcm':pcm,'window':asdict(item.segment),
                           'backend':identity,'request':asdict(request)})
            entry=cache.get_json(key)
            value=None
            if entry is not None and entry.get('schema')=='transcription-request-v1':
                try:
                    value=transcription_segment_from_dict(entry['value'])
                    target=value.backend_metadata.get('distribution_file')
                    if target is not None:
                        from pathlib import Path
                        if not Path(target).is_file():value=None
                except (ValueError,TypeError,KeyError):value=None
            if value is None:
                output=transcriber.transcribe((item,),request=request)
                if len(output)!=1:raise ValueError('one result per request required')
                value=_bind(item,output[0],request,request.context)
                checkpoint(cancel)
                cache.set_json(key,{'schema':'transcription-request-v1','value':asdict(value)})
            values.append(value);completed+=1
            checkpoint(cancel,event_sink,'transcription',completed,len(inputs))
        return tuple(values)
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
