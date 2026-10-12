"""歌词搜索使用的无网络文本变体与元数据评分。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from rapidfuzz import fuzz
from opencc import OpenCC

from .normalize import SEARCH_SPECIAL_CHAR_REPLACEMENTS, normalize_text

_SPACES = re.compile(r"\s+")
_SEARCH_BRACKETS = re.compile(r"[\(\[（【].*?[\)\]）】]")
_SEARCH_PUNCTUATION = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\s'-]+")


@dataclass(frozen=True, slots=True)
class SearchScoreFields:
    duration: float | None
    title_candidates: tuple[str, ...]
    artist_candidates: tuple[str, ...]
    album_candidates: tuple[str, ...]


@cache
def _traditional_to_simplified() -> OpenCC:
    """仅在实际评分时加载 OpenCC 词典，基础导入不初始化词典。"""
    return OpenCC("t2s")


def combined_fuzzy_score(left: str, right: str, *, full_match_weight: float = 0.4) -> float:
    if not 0 <= full_match_weight <= 1:
        raise ValueError("full_match_weight 必须在 [0, 1] 内")
    partial = fuzz.partial_ratio(left, right)
    full = fuzz.ratio(left, right)
    return float(partial * (1 - full_match_weight) + full * full_match_weight)


def normalize_search_text(text: str | None) -> str:
    return normalize_text(
        text,
        keep_spaces=True,
        replacements=SEARCH_SPECIAL_CHAR_REPLACEMENTS,
        normalize_dash=True,
    )


def normalize_score_text(text: str | None) -> str:
    value = normalize_search_text(text)
    if not value:
        return ""
    return _SPACES.sub(" ", _traditional_to_simplified().convert(value)).strip()


def strip_search_bracket_suffix(text: str | None) -> str:
    value = normalize_search_text(text)
    return _SPACES.sub(" ", _SEARCH_BRACKETS.sub(" ", value)).strip(" ._-")


def simplify_search_text(text: str | None) -> str:
    value = strip_search_bracket_suffix(text)
    return _SPACES.sub(" ", _SEARCH_PUNCTUATION.sub(" ", value)).strip(" ._-")


def build_search_text_variants(text: str | None) -> list[str]:
    variants: list[str] = []
    for candidate in (
        str(text or "").strip(),
        normalize_search_text(text),
        strip_search_bracket_suffix(text),
        simplify_search_text(text),
    ):
        value = _SPACES.sub(" ", candidate).strip()
        if value and value not in variants:
            variants.append(value)
    return variants


def score_search_candidate(
    *,
    title: str,
    duration: float,
    fields: SearchScoreFields,
    artist: str | None = None,
    album: str | None = None,
    name_weight: float = 0.7,
    album_weight: float = 0.2,
    artist_weight: float = 0.1,
    name_full_match_weight: float = 0.4,
    artist_full_match_weight: float = 0.2,
    album_full_match_weight: float = 0.3,
    duration_threshold: float = 15.0,
) -> float:
    """保留旧来源的加权逻辑；时长不符的候选不参与后续内容重排。"""
    if fields.duration is None or abs(float(fields.duration) - duration) > duration_threshold:
        return 0.0

    score_title = normalize_score_text(title)
    score_artist = normalize_score_text(artist)
    score_album = normalize_score_text(album)
    titles = tuple(filter(None, map(normalize_score_text, fields.title_candidates)))
    artists = tuple(filter(None, map(normalize_score_text, fields.artist_candidates)))
    albums = tuple(filter(None, map(normalize_score_text, fields.album_candidates)))
    title_score = max(
        (combined_fuzzy_score(score_title, value, full_match_weight=name_full_match_weight) for value in titles),
        default=0.0,
    ) if score_title else 0.0
    artist_score = max(
        (combined_fuzzy_score(score_artist, value, full_match_weight=artist_full_match_weight) for value in artists),
        default=0.0,
    ) if score_artist and artist is not None else 0.0
    album_score = max(
        (combined_fuzzy_score(score_album, value, full_match_weight=album_full_match_weight) for value in albums),
        default=0.0,
    ) if score_album and album is not None else 0.0

    effective_artist_weight = artist_weight if artist is not None else 0.0
    effective_album_weight = album_weight if album is not None else 0.0
    total_weight = name_weight + effective_artist_weight + effective_album_weight
    if total_weight <= 0:
        return 0.0
    return float(
        (name_weight * title_score + effective_artist_weight * artist_score + effective_album_weight * album_score)
        / total_weight
    )
