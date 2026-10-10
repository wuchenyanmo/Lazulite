"""按需加载的模型后端。"""

from .qwen import QwenBackend, QwenConfig
from .whisper import WhisperBackend, WhisperConfig

__all__ = ["QwenBackend", "QwenConfig", "WhisperBackend", "WhisperConfig"]
