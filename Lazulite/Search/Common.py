from __future__ import annotations

import re
from dataclasses import dataclass

from fuzzywuzzy import fuzz
from Lazulite.TextNormalize import (
    RE_DASH_VARIANTS,
    RE_SPACES,
    SEARCH_SPECIAL_CHAR_REPLACEMENTS,
    normalize_text,
)

RE_SEARCH_BRACKETS = re.compile(r"[\(\[（【].*?[\)\]）】]")
RE_SEARCH_PUNCT = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\s'-]+")


@dataclass(slots=True)
class SearchScoreFields:
    duration: float | None
    title_candidates: list[str]
    artist_candidates: list[str]
    album_candidates: list[str]


def combined_fuzzy_score(str1: str, str2: str, full_match_weight: float = 0.4) -> float:
    partial = fuzz.partial_ratio(str1, str2)
    full = fuzz.ratio(str1, str2)
    return float(partial * (1 - full_match_weight) + full * full_match_weight)


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
    candidate_duration = fields.duration
    if candidate_duration is None:
        return 0.0
    if abs(float(candidate_duration) - duration) > duration_threshold:
        return 0.0

    score = {
        "name": (
            max(combined_fuzzy_score(title, item, full_match_weight=name_full_match_weight) for item in fields.title_candidates)
            if fields.title_candidates else 0.0
        ),
    }

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        score["artist"] = (
            max(combined_fuzzy_score(artist, item, full_match_weight=artist_full_match_weight) for item in fields.artist_candidates)
            if fields.artist_candidates else 0.0
        )

    if album is None:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        score["album"] = (
            max(combined_fuzzy_score(album, item, full_match_weight=album_full_match_weight) for item in fields.album_candidates)
            if fields.album_candidates else 0.0
        )

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    value = (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight
    return float(value)


def normalize_search_text(text: str | None) -> str:
    return normalize_text(
        text,
        keep_spaces=True,
        replacements=SEARCH_SPECIAL_CHAR_REPLACEMENTS,
        normalize_dash=True,
    )


def strip_search_bracket_suffix(text: str | None) -> str:
    value = normalize_search_text(text)
    if not value:
        return ""
    value = RE_SEARCH_BRACKETS.sub(" ", value)
    value = RE_SPACES.sub(" ", value).strip(" ._-")
    return value


def simplify_search_text(text: str | None) -> str:
    value = strip_search_bracket_suffix(text)
    if not value:
        return ""
    value = RE_SEARCH_PUNCT.sub(" ", value)
    value = RE_SPACES.sub(" ", value).strip(" ._-")
    return value


def build_search_text_variants(text: str | None) -> list[str]:
    variants: list[str] = []
    for candidate in [
        str(text or "").strip(),
        normalize_search_text(text),
        strip_search_bracket_suffix(text),
        simplify_search_text(text),
    ]:
        value = RE_SPACES.sub(" ", candidate).strip()
        if value and value not in variants:
            variants.append(value)
    return variants
