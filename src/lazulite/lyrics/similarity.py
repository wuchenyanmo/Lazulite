"""搜索和行级对齐共用的文本相似度函数。"""

from __future__ import annotations

from difflib import SequenceMatcher

from .normalize import normalize_lyric_text


def text_similarity(left: str | None, right: str | None) -> float:
    """返回两个歌词匹配视图的 [0, 1] 相似度；空正文没有匹配资格。"""
    normalized_left = normalize_lyric_text(left)
    normalized_right = normalize_lyric_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    ratio = SequenceMatcher(None, normalized_left, normalized_right).ratio()

    def grams(text, size):
        return (
            {text[i : i + size] for i in range(len(text) - size + 1)}
            if len(text) >= size
            else {text}
        )

    ua, ub = grams(normalized_left, 1), grams(normalized_right, 1)
    ba, bb = grams(normalized_left, 2), grams(normalized_right, 2)
    return (
        0.40 * ratio
        + 0.22 * len(ua & ub) / max(len(ua | ub), 1)
        + 0.26 * len(ba & bb) / max(len(ba | bb), 1)
        + 0.12 * min(len(ua & ub) / max(len(ua), 1), len(ua & ub) / max(len(ub), 1))
    )


def prefix_similarity(
    left: str | None, right: str | None, *, prefix_chars: int = 12
) -> float:
    """比较歌词前缀，用于重复副歌的锚点质量诊断。"""
    if prefix_chars < 1:
        raise ValueError("prefix_chars 必须为正数")
    normalized_left = normalize_lyric_text(left)
    normalized_right = normalize_lyric_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    size = min(len(normalized_left), len(normalized_right), prefix_chars)
    return SequenceMatcher(
        None, normalized_left[:size], normalized_right[:size]
    ).ratio()
