"""无模型的组合边界测试。"""

from __future__ import annotations

import numpy as np
import pytest

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.audio.activity import VocalConfig
from lazulite.models.transcription import BackendInfo, TranscriptionSegment
from lazulite.transcription.base import (
    TranscriberCapabilities,
    TranscriptionInput,
    TranscriptionRequest,
)
from lazulite.transcription.run import transcribe_segments
from lazulite.pipeline import AlignmentPipeline, PipelineConfig
from lazulite.models.lyrics import LyricDocument, LyricLine
from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult
import lazulite.pipeline as pipeline_module


class _RecordingBackend:
    info = BackendInfo("fake", "fake/model", "test", "float32")
    capabilities = TranscriberCapabilities("none", context=False, max_audio_seconds=4.0)

    def __init__(self) -> None:
        self.inputs: tuple[TranscriptionInput, ...] = ()

    def transcribe(self, inputs, *, request: TranscriptionRequest):
        self.inputs = tuple(inputs)
        return tuple(
            TranscriptionSegment(
                source_segment_id=item.segment.id,
                text=f"window-{index}",
                language=request.language,
                start=None,
                end=None,
                confidence=None,
                confidence_kind=None,
                timed_text=(),
                backend_metadata={},
            )
            for index, item in enumerate(inputs)
        )

    def close(self) -> None:
        pass


def test_transcription_runner_splits_long_windows_without_changing_source_id() -> None:
    audio = AudioBuffer(np.zeros((1, 12_000), dtype=np.float32), 1_000, "audio")
    segment = VocalSegment("verse-1", None, 0.0, 10.0, 1.0, 9.0, 0.5, 0.1)
    backend = _RecordingBackend()

    result = transcribe_segments(
        audio, (segment,), transcriber=backend, request=TranscriptionRequest(language="ja")
    )

    assert [(item.segment.start, item.segment.end) for item in backend.inputs] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 10.0)
    ]
    assert [item.source_segment_id for item in result.segments] == ["verse-1"] * 3
    assert all(item.start is None for item in result.segments)




def test_pipeline_injects_backend_without_writing_audio():
    audio = AudioBuffer(np.ones((1, 8_000), dtype=np.float32) * 0.5, 1_000, "audio")
    backend = _RecordingBackend(); pipeline = AlignmentPipeline(transcriber=backend)
    result = pipeline.run(audio, LyricDocument((LyricLine("L1", "window-0"),)), config=PipelineConfig(mode="dp", vocal=VocalConfig(min_segment_seconds=0.2)))
    assert result.transcription is not None
    assert audio.samples[0, 0] == 0.5


def test_pipeline_keeps_all_lines_unresolved_when_no_vocals_are_detected() -> None:
    audio = AudioBuffer(np.zeros((1, 1_000), dtype=np.float32), 1_000, "silent")
    result = AlignmentPipeline(transcriber=_RecordingBackend()).run(
        audio,
        LyricDocument((LyricLine("L1", "第一行"), LyricLine("L2", "第二行"))),
    )

    assert result.transcription is None
    assert [line.line_id for line in result.alignment.lines] == ["L1", "L2"]
    assert all(line.timing_source == "unresolved" for line in result.alignment.lines)




def test_explicit_offset_requires_complete_source_times_before_running_models() -> None:
    audio = AudioBuffer(np.zeros((1, 1_000), dtype=np.float32), 1_000, "audio")

    with pytest.raises(ValueError, match="source_start"):
        AlignmentPipeline(transcriber=_RecordingBackend()).run(
            audio, LyricDocument((LyricLine("L1", "歌词"),)), config=PipelineConfig(mode="offset")
        )


def test_hybrid_refinement_receives_the_dp_match_result(monkeypatch) -> None:
    dp_result = LineAlignmentResult((
        AlignedLyricLine("L1", 1.0, 2.0, 1.0, 1.0, "acoustic_core", ("vocal-0",)),
    ), "dp", True, ())
    refined = LineAlignmentResult((
        AlignedLyricLine("L1", 2.0, 3.0, 1.0, 1.0, "source_offset", ("vocal-0",)),
    ), "offset", True, ())
    received = []
    monkeypatch.setattr(pipeline_module, "align_lines", lambda *args, **kwargs: dp_result)
    monkeypatch.setattr(pipeline_module, "align_with_offset", lambda *args, **kwargs: received.append(kwargs["guidance"]) or refined)
    audio = AudioBuffer(np.ones((1, 8_000), dtype=np.float32) * 0.5, 1_000, "audio")

    result = AlignmentPipeline(transcriber=_RecordingBackend()).run(
        audio,
        LyricDocument((LyricLine("L1", "window-0", 0.0, 1.0),)),
        config=PipelineConfig(mode="hybrid", vocal=VocalConfig(min_segment_seconds=0.2)),
    )

    assert received == [dp_result]
    assert result.alignment.strategy == "hybrid"






def test_explicit_stages_skip_backend_and_invalid_windows_fail(tmp_path):
    from dataclasses import replace
    from lazulite.exchange import write_transcriptions,read_transcriptions
    audio=AudioBuffer(np.ones((1,8000),np.float32)*.5,1000,'audio')
    doc=LyricDocument((LyricLine('L1','window-0'),))
    cfg=PipelineConfig(mode='dp',vocal=VocalConfig(min_segment_seconds=.2))
    first=AlignmentPipeline(transcriber=_RecordingBackend()).run(audio,doc,config=cfg)
    path=tmp_path/'transcriptions.json'
    write_transcriptions(path,first.transcriptions)
    supplied=read_transcriptions(path)
    second=AlignmentPipeline().run(audio,doc,config=cfg,transcriptions=supplied)
    assert second.alignment==first.alignment and second.transcription==first.transcription
    stage=supplied['full']
    bad=replace(stage,result=replace(stage.result,segments=stage.result.segments[:-1]))
    with pytest.raises(ValueError,match='missing request windows'):
        AlignmentPipeline(transcriber=_RecordingBackend()).run(audio,doc,config=cfg,transcriptions={'full':bad})
    with pytest.raises(ValueError,match='conditions'):
        AlignmentPipeline().run(audio,doc,config=replace(cfg,language='en'),transcriptions=supplied)


def test_auto_keeps_preselection_and_full_timing_conditions_separate(monkeypatch):
    from dataclasses import replace
    class Fine(_RecordingBackend):
        capabilities=TranscriberCapabilities('subword',True,4)
    rejected=LineAlignmentResult((), 'offset',False,())
    monkeypatch.setattr(pipeline_module,'align_with_offset',lambda *args,**kw:rejected)
    audio=AudioBuffer(np.ones((1,8000),np.float32)*.5,1000,'audio')
    doc=LyricDocument((LyricLine('L1','window-0',0.,1.),))
    cfg=PipelineConfig(mode='auto',vocal=VocalConfig(min_segment_seconds=.2))
    first=AlignmentPipeline(transcriber=Fine()).run(audio,doc,config=cfg)
    stages=first.transcriptions
    assert set(stages)=={'preselected','full'}
    assert stages['preselected'].request.include_timing is False
    assert stages['full'].request.include_timing is True
    assert AlignmentPipeline().run(audio,doc,config=cfg,transcriptions=stages).alignment==first.alignment
    with pytest.raises(ValueError,match='missing transcription stage: full'):
        AlignmentPipeline().run(audio,doc,config=cfg,transcriptions={'preselected':stages['preselected']})
    with pytest.raises(ValueError,match='conditions'):
        AlignmentPipeline().run(audio,doc,config=cfg,transcriptions={'preselected':stages['full']})
