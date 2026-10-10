"""在线歌词来源的接口与适配器。"""

from .base import FetchedLyricCandidate, LyricProvider, SearchCandidate
from .netease import NeteaseProvider
from .qqmusic import QQMusicProvider
from .lrclib import LRCLIBProvider
from .kugou import KugouProvider

__all__ = ["FetchedLyricCandidate", "KugouProvider", "LRCLIBProvider", "LyricProvider", "NeteaseProvider", "QQMusicProvider", "SearchCandidate"]
