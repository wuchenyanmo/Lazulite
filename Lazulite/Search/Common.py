from __future__ import annotations

import re

from fuzzywuzzy import fuzz
from Lazulite.TextNormalize import (
    RE_DASH_VARIANTS,
    RE_SPACES,
    SEARCH_SPECIAL_CHAR_REPLACEMENTS,
    normalize_text,
)

RE_SEARCH_BRACKETS = re.compile(r"[\(\[（【].*?[\)\]）】]")
RE_SEARCH_PUNCT = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\s'-]+")


def combined_fuzzy_score(str1: str, str2: str, full_match_weight: float = 0.4) -> float:
    partial = fuzz.partial_ratio(str1, str2)
    full = fuzz.ratio(str1, str2)
    return float(partial * (1 - full_match_weight) + full * full_match_weight)


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
