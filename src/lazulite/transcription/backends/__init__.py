"""按需加载的模型后端。"""

from .whisper import WhisperBackend, WhisperConfig

__all__ = ["WhisperBackend", "WhisperConfig"]
