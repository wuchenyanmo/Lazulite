from __future__ import annotations

from typing import Protocol

from lazulite.models.alignment import ForcedAlignmentRequest, ForcedAlignmentResult
from lazulite.models.audio import AudioBuffer
from lazulite.models.transcription import BackendInfo


class ForcedAligner(Protocol):
    info: BackendInfo
    def align(self, audio: AudioBuffer, *, request: ForcedAlignmentRequest) -> ForcedAlignmentResult: ...
    def close(self) -> None: ...
