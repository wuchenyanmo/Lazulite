"""Lazulite 的轻量公共类型。

具体模型后端按需从 ``lazulite.transcription.backends`` 导入，避免基础安装
在导入本包时加载 Torch、Demucs 或任何模型权重。
"""

from .models.alignment import (
    AlignedLyricLine,
    ForcedAlignmentRequest,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    LineAlignmentResult,
)
from .models.audio import AudioBuffer, VocalAnalysis, VocalSegment
from .models.lyrics import LyricDocument, LyricLine, ReadingSpan
from .models.transcription import (
    BackendInfo,
    TimedText,
    TranscriptionResult,
    TranscriptionSegment,
)

__version__ = "0.1.0"

__all__ = [
    "AlignedLyricLine",
    "AudioBuffer",
    "BackendInfo",
    "ForcedAlignmentRequest",
    "ForcedAlignmentResult",
    "ForcedAlignmentUnit",
    "LineAlignmentResult",
    "LyricDocument",
    "LyricLine",
    "ReadingSpan",
    "TimedText",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VocalAnalysis",
    "VocalSegment",
]
