from __future__ import annotations

import re
import unicodedata
from typing import Iterable

RE_SPACES = re.compile(r"\s+")
RE_DASH_VARIANTS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")

BASE_TEXT_REPLACEMENTS = {
    "’": "'",
    "`": "'",
    "“": '"',
    "”": '"',
}
SEARCH_SPECIAL_CHAR_REPLACEMENTS = {
    "☆": " ",
    "★": " ",
    "～": " ",
    "~": " ",
}
KUGOU_ARTIST_SEPARATORS = ("、", " / ", "/", "&", "·", ";", "；", ",", "，")


def apply_text_replacements(text: str, replacements: dict[str, str]) -> str:
    value = text
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def normalize_text(
    text: str | None,
    *,
    keep_spaces: bool = True,
    replacements: dict[str, str] | None = None,
    normalize_dash: bool = False,
) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = apply_text_replacements(value, BASE_TEXT_REPLACEMENTS)
    if replacements:
        value = apply_text_replacements(value, replacements)
    if normalize_dash:
        value = RE_DASH_VARIANTS.sub("-", value)
    value = value.strip()
    if keep_spaces:
        return RE_SPACES.sub(" ", value)
    return "".join(value.split())


def clean_text(value: object | None) -> str:
    return str(value or "").strip()


def unique_non_empty_texts(values: Iterable[object | None]) -> list[str]:
    results: list[str] = []
    for item in values:
        text = clean_text(item)
        if text and text not in results:
            results.append(text)
    return results


def split_text(value: str | None, separators: Iterable[str]) -> list[str]:
    normalized = str(value or "")
    for sep in separators:
        normalized = normalized.replace(sep, "|")
    return unique_non_empty_texts(normalized.split("|"))
