"""无模型的组合边界测试。"""

from __future__ import annotations

import numpy as np
from lazulite.cache import FileCache, cache_key

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


class _RecordingBackend:
    info = BackendInfo("fake", "fake/model", "test", "test", "float32")
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

    @property
    def cache_identity(self):
        return {"backend": {"name": "fake", "revision": "test"}, "recipe": "test-v1"}


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


def test_file_cache_uses_deterministic_keys(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key = cache_key({"stage": "test", "value": [1, 2]})
    cache.set_json(key, {"ok": True})
    assert cache.get_json(key) == {"ok": True}
    assert key == cache_key({"value": [1, 2], "stage": "test"})


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


def test_pipeline_reuses_complete_result_only_for_a_content_addressed_cache_key(tmp_path) -> None:
    audio = AudioBuffer(np.ones((1, 1_000), dtype=np.float32) * 0.5, 1_000, "sha256:audio")
    document = LyricDocument((LyricLine("L1", "window-0"),))
    config = PipelineConfig(mode="dp", vocal=VocalConfig(min_segment_seconds=0.2))
    cache = FileCache(tmp_path)
    first = _RecordingBackend()
    second = _RecordingBackend()

    AlignmentPipeline(transcriber=first, cache=cache).run(audio, document, config=config)
    cached = AlignmentPipeline(transcriber=second, cache=cache).run(audio, document, config=config)

    assert first.inputs
    assert second.inputs == ()
    assert cached.alignment.lines[0].line_id == "L1"
