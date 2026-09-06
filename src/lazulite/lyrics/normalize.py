"""歌词与检索共用的纯文本规范化函数。"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

_SPACES = re.compile(r"\s+")
_DASH_VARIANTS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_FURIGANA = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶ]+)\(([\u3040-\u30ffー・ ]+)\)")
_LYRIC_QUOTES = re.compile(r"[「」『』【】〈〉《》〔〕〘〙〚〛“”‘’\"]+")
_LYRIC_MARKS = re.compile(r"[·•♪♬♫]+")
_LYRIC_PUNCTUATION = re.compile(r"[，。！？、,.!?;；:：~～…]+")

BASE_TEXT_REPLACEMENTS: dict[str, str] = {
    "’": "'",
    "`": "'",
    "“": '"',
    "”": '"',
}
SEARCH_SPECIAL_CHAR_REPLACEMENTS: dict[str, str] = {
    "☆": " ",
    "★": " ",
    "～": " ",
    "~": " ",
}
KUGOU_ARTIST_SEPARATORS = ("、", " / ", "/", "&", "·", ";", "；", ",", "，")


def normalize_text(
    text: object | None,
    *,
    keep_spaces: bool = True,
    replacements: Mapping[str, str] | None = None,
    normalize_dash: bool = False,
) -> str:
    """做通用 Unicode 规范化，不改变 ``LyricLine.text`` 中保存的原文。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    for source, target in BASE_TEXT_REPLACEMENTS.items():
        value = value.replace(source, target)
    if replacements:
        for source, target in replacements.items():
            value = value.replace(source, target)
    if normalize_dash:
        value = _DASH_VARIANTS.sub("-", value)
    value = value.strip()
    return _SPACES.sub(" ", value) if keep_spaces else "".join(value.split())


def normalize_lyric_text(text: object | None) -> str:
    """生成用于匹配的视图；日语原文和括号读音均不写回输入歌词。"""
    value = normalize_text(text)
    value = _FURIGANA.sub(r"\1", value)
    value = _LYRIC_QUOTES.sub(" ", value)
    value = _LYRIC_MARKS.sub(" ", value)
    value = _LYRIC_PUNCTUATION.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


def clean_text(value: object | None) -> str:
    return str(value or "").strip()


def unique_non_empty_texts(values: Iterable[object | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in result:
            result.append(text)
    return result


def split_text(value: str | None, separators: Iterable[str]) -> list[str]:
    normalized = str(value or "")
    for separator in separators:
        normalized = normalized.replace(separator, "|")
    return unique_non_empty_texts(normalized.split("|"))
