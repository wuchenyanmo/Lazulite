"""音频 I/O、分离与活动分析。"""

from .io import decode_audio, read_audio_metadata
from .activity import VocalConfig, analyze_vocals

__all__ = ["VocalConfig", "analyze_vocals", "decode_audio", "read_audio_metadata"]
