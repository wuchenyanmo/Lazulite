"""MiliKara MIT kana/mora routines, ff2d2f54706ecb140bc5b275037eff64187ef6ef. See THIRD_PARTY_LICENSES."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


SMALL_MERGE = set("ゃゅょぁぃぅぇぉゎ")
# wave dash 〜 (U+301C) and full-width tilde ～ (U+FF5E) are written for a long vowel as often as ー
LONG_MARKS = set("ー〜～")
# the ASCII tilde is only a long vowel after kana (``ラララ~``); elsewhere it is punctuation
ASCII_TILDE = "~"
SOKUON = "っ"
HATSUON = "ん"


@dataclass
class Mora:
    text: str
    flags: list[str] = field(default_factory=list)


def to_hiragana(s: str) -> str:
    """Katakana -> hiragana (ヴ -> ゔ); long-vowel marks -> ``ー``; other characters unchanged.

    〜 / ～ always become ``ー`` (NFKC would fold ～ to an ASCII ``~`` that nothing reads as a
    long vowel); an ASCII ``~`` does when it follows kana, or when the string is only tildes
    (a single long-vowel unit such as ``["ら", "~"]``).
    """
    s = unicodedata.normalize("NFKC", s.replace("～", "ー").replace("〜", "ー"))
    only_marks = bool(s) and all(c in "ー" + ASCII_TILDE for c in s)
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == ASCII_TILDE and (only_marks or (out and _kana_code(out[-1]))):
            out.append("ー")
        else:
            out.append(ch)
    return "".join(out)


def _kana_code(ch: str) -> bool:
    code = ord(ch)
    return (0x3041 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF) and ch not in _NON_KANA


# characters inside the kana Unicode blocks that are punctuation / marks,
# not pronounced: ゠ (double hyphen), ・ (middle dot, e.g. "・・・"),
# standalone ゛ ゜
_NON_KANA = {"\u30a0", "\u30fb", "\u309b", "\u309c"}


def is_kana(ch: str) -> bool:
    code = ord(ch)
    if ch in _NON_KANA:
        return False
    return 0x3041 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF or ch in LONG_MARKS


def is_kanji(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF
            or ch in "々〆ヶ")


def is_kana_text(s: str) -> bool:
    """Only kana and long-vowel marks (an ASCII ``~`` counts after kana: ``ラララ~``)."""
    if not s:
        return False
    for i, c in enumerate(s):
        if not (is_kana(c) or (c == ASCII_TILDE and i > 0 and (is_kana(s[i - 1]) or s[i - 1] == ASCII_TILDE))):
            return False
    return True


def is_long_mark_text(s: str) -> bool:
    """Only long-vowel marks (ー 〜 ～ ~): a separate "word" that lengthens the one before it."""
    return bool(s) and all(c in LONG_MARKS or c == ASCII_TILDE for c in s)


def split_morae(kana: str) -> list[Mora]:
    """Split a kana string (hiragana or katakana) into morae."""
    h = to_hiragana(kana)
    morae: list[Mora] = []
    for ch in h:
        if ch in SMALL_MERGE and morae and not morae[-1].flags:
            morae[-1].text += ch
        elif ch == SOKUON:
            morae.append(Mora(ch, ["sokuon"]))
        elif ch == HATSUON:
            morae.append(Mora(ch, ["hatsuon"]))
        elif ch in LONG_MARKS or ch == ASCII_TILDE:
            morae.append(Mora("ー", ["long"]))
        else:
            morae.append(Mora(ch, []))
    return morae


