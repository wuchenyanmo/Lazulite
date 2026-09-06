"""在线歌词来源的接口与适配器。"""

from .base import FetchedLyricCandidate, LyricProvider, SearchCandidate
from .netease import NeteaseProvider

__all__ = ["FetchedLyricCandidate", "LyricProvider", "NeteaseProvider", "SearchCandidate"]
