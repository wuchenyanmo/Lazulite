"""模型无关转写协议与调度。"""

from .base import (
    Transcriber,
    TranscriberCapabilities,
    TranscriptionInput,
    TranscriptionRequest,
)
from .run import TranscriptionConfig, transcribe_segments

__all__ = [
    "Transcriber",
    "TranscriptionConfig",
    "TranscriberCapabilities",
    "TranscriptionInput",
    "TranscriptionRequest",
    "transcribe_segments",
]
