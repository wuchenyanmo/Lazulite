"""在线歌词来源的接口与适配器。"""

from .base import FetchedLyricCandidate, LyricProvider, SearchCandidate
from .netease import NeteaseProvider
from .qqmusic import QQMusicProvider
from .lrclib import LRCLIBProvider

__all__ = ["FetchedLyricCandidate", "LRCLIBProvider", "LyricProvider", "NeteaseProvider", "QQMusicProvider", "SearchCandidate"]
