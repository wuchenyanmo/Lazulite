"""无模型的组合边界测试。"""

from __future__ import annotations

import numpy as np

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.transcription import BackendInfo, TranscriptionSegment
from lazulite.transcription.base import (
    TranscriberCapabilities,
    TranscriptionInput,
    TranscriptionRequest,
)
from lazulite.transcription.run import transcribe_segments


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
