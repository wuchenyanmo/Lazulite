"""模型无关转写协议与调度。"""

from .base import (
    Transcriber,
    TranscriberCapabilities,
    TranscriptionInput,
    TranscriptionRequest,
)
from .run import transcribe_segments

__all__ = [
    "Transcriber",
    "TranscriberCapabilities",
    "TranscriptionInput",
    "TranscriptionRequest",
    "transcribe_segments",
]
