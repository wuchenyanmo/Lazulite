"""歌词输入、规范化与搜索能力。"""

from .lrc import clean_translation, LrcDocument, read_lrc, read_plain_text, with_translation, write_lrc
from .normalize import normalize_lyric_text, normalize_text
from .similarity import prefix_similarity, text_similarity
from .service import (
    RankedLyricCandidate,
    build_default_providers,
    fetch_lyrics,
    rank_lyrics,
    search_lyrics,
)
from .ttml import TtmlFormatError, read_ttml_for_alignment

__all__ = [
    "LrcDocument",
    "RankedLyricCandidate",
    "build_default_providers",
    "clean_translation",
    "fetch_lyrics",
    "normalize_lyric_text",
    "normalize_text",
    "prefix_similarity",
    "read_lrc",
    "read_plain_text",
    "read_ttml_for_alignment",
    "rank_lyrics",
    "search_lyrics",
    "TtmlFormatError",
    "text_similarity",
    "with_translation",
    "write_lrc",
]
