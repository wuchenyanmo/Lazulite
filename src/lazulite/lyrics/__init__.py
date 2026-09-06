"""歌词输入、规范化与搜索能力。"""

from .lrc import LrcDocument, read_lrc, read_plain_text, with_translation, write_lrc
from .normalize import normalize_lyric_text, normalize_text
from .ttml import TtmlFormatError, read_ttml_for_alignment

__all__ = [
    "LrcDocument",
    "normalize_lyric_text",
    "normalize_text",
    "read_lrc",
    "read_plain_text",
    "read_ttml_for_alignment",
    "TtmlFormatError",
    "with_translation",
    "write_lrc",
]
