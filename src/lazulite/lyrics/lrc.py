"""LRC 与纯文本歌词的输入、翻译关联和显式导出。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lazulite.models.lyrics import LyricDocument, LyricLine

from .normalize import normalize_lyric_text, normalize_text

_METADATA = re.compile(r"^\[([A-Za-z]+):(.*)\]$")
_TIMESTAMP = re.compile(r"^\[(\d{2,}:\d{2}\.\d{2,3})\](.*)$")
_METADATA_SEPARATOR = re.compile(r"\s*[:：|｜]\s*")
_SINGER_COLON = re.compile(r"^(?P<label>[^\s:：|｜]{1,24})\s*[:：|｜]\s*(?P<body>.+)$")
_SINGER_BRACKET = re.compile(
    r"^[\(\[（【「『<＜](?P<label>.{1,24}?)[\)\]）】」』>＞]\s*(?P<body>.+)$"
)
_SINGER_DASH = re.compile(
    r"^(?P<label>[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff/&+・ ]{1,24})\s*[-~]\s*(?P<body>.+)$"
)

_DEFAULT_METADATA_KEYS = {"ti", "ar", "al", "by", "offset", "la", "id", "length", "re", "ve"}
_METADATA_LABELS = {
    "作词", "作詞", "词", "詞", "作曲", "曲", "编曲", "編曲", "作词作曲", "词曲",
    "演唱", "歌手", "原唱", "vocal", "artist", "title", "标题", "標題", "album",
    "专辑", "專輯", "制作", "製作", "producer", "lyrics", "music", "composer",
    "lyricist", "chorus", "和声", "和聲", "mix", "mixed", "master", "录音", "錄音",
}
_SINGER_ROLE_WORDS = {
    "男", "女", "合", "和", "齐", "齊", "全", "全员", "全員", "duet", "solo",
    "chorus", "rap", "lead", "main", "vocal", "和声", "和聲",
}


@dataclass(frozen=True, slots=True)
class LrcDocument:
    """LRC 的解析结果；歌词正文仍通过 ``LyricDocument`` 统一进入算法。"""

    document: LyricDocument
    metadata: tuple[tuple[str, str], ...] = ()
    translations: tuple[tuple[str, str], ...] = ()

    def translation_for(self, line_id: str) -> str | None:
        return dict(self.translations).get(line_id)


def _parse_timestamp(value: str) -> float:
    minute, second = value.split(":", maxsplit=1)
    return 60 * float(minute) + float(second)


def _looks_like_metadata_label(label: str) -> bool:
    compact = normalize_text(label, keep_spaces=False).lower()
    return compact in _DEFAULT_METADATA_KEYS or compact in {
        item.lower() for item in _METADATA_LABELS
    }


def _split_key_value(text: str) -> tuple[str, str] | None:
    parts = _METADATA_SEPARATOR.split(text, maxsplit=1)
    if len(parts) != 2:
        return None
    key, value = (part.strip() for part in parts)
    return (key, value) if key and value else None


def _metadata_from_timestamp_line(timestamp: float, text: str) -> tuple[str, str] | None:
    """只把明确的头部信息当元数据，避免吞掉正常歌词。"""
    key_value = _split_key_value(text)
    if key_value is None:
        return None
    key, value = key_value
    if _looks_like_metadata_label(key):
        return key, value
    if timestamp <= 20 and len(key) <= 12 and len(value) <= 40:
        return key, value
    return None


def _looks_like_singer_label(label: str) -> bool:
    clean = normalize_text(label)
    compact = normalize_text(label, keep_spaces=False)
    if not clean or len(clean) > 24 or _looks_like_metadata_label(clean):
        return False
    if compact.lower() in {word.lower() for word in _SINGER_ROLE_WORDS}:
        return True
    return bool(
        re.fullmatch(r"[A-Za-z0-9/&+]+", compact)
        or re.fullmatch(r"[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9/&+・ ]+", clean)
    )


def _extract_singer_marker(text: str) -> tuple[str | None, str]:
    for pattern in (_SINGER_BRACKET, _SINGER_COLON, _SINGER_DASH):
        match = pattern.match(text)
        if match is None:
            continue
        label = normalize_text(match.group("label"))
        body = match.group("body").strip()
        if body and _looks_like_singer_label(label):
            return label, body
    return None, text


def read_plain_text(text: str, *, language: str | None = None) -> LyricDocument:
    """按输入顺序生成稳定字符串 ID，不伪造时间。"""
    lines: list[LyricLine] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(LyricLine(id=f"line-{len(lines)}", text=line))
    return LyricDocument(lines=tuple(lines), language=language)


def read_lrc(text: str, *, language: str | None = None) -> LrcDocument:
    """解析 LRC 的元数据、行时间与歌手前缀，保留歌词正文原样。"""
    metadata: list[tuple[str, str]] = []
    lines: list[LyricLine] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        metadata_match = _METADATA.match(line)
        if metadata_match is not None:
            key, value = metadata_match.groups()
            metadata.append((normalize_text(key, keep_spaces=False).lower(), normalize_text(value)))
            continue
        lyric_match = _TIMESTAMP.match(line)
        if lyric_match is None:
            continue
        timestamp_text, raw_text = lyric_match.groups()
        timestamp = _parse_timestamp(timestamp_text)
        clean_text = normalize_text(raw_text)
        timestamp_metadata = _metadata_from_timestamp_line(timestamp, clean_text)
        if timestamp_metadata is not None:
            key, value = timestamp_metadata
            metadata.append((normalize_text(key, keep_spaces=False).lower(), value))
            continue
        singer, body = _extract_singer_marker(clean_text)
        if not normalize_lyric_text(body):
            continue
        lines.append(
            LyricLine(
                id=f"line-{len(lines)}",
                text=body,
                source_start=timestamp,
                singer=singer,
            )
        )
    return LrcDocument(LyricDocument(tuple(lines), language=language), tuple(metadata))


def with_translation(document: LrcDocument, translation_lrc: str) -> LrcDocument:
    """按最近真实时间关联翻译；无时间的翻译按行序关联。"""
    translation = read_lrc(translation_lrc, language=document.document.language)
    target_lines = document.document.lines
    source_lines = translation.document.lines
    if not target_lines or not source_lines:
        return document

    mappings: list[tuple[str, str]] = []
    timed_target = all(line.source_start is not None for line in target_lines)
    timed_source = all(line.source_start is not None for line in source_lines)
    if timed_target and timed_source:
        for source in source_lines:
            assert source.source_start is not None
            matched = min(
                target_lines,
                key=lambda target: abs((target.source_start or 0.0) - source.source_start),
            )
            mappings.append((matched.id, source.text))
    else:
        mappings.extend((target.id, source.text) for target, source in zip(target_lines, source_lines))
    return LrcDocument(document.document, document.metadata, tuple(mappings))


def write_lrc(
    document: LrcDocument,
    *,
    include_translations: bool = False,
    translation_brackets: str = "【】",
) -> str:
    """显式导出 LRC；没有真实行起点时拒绝伪造时间轴。"""
    if len(translation_brackets) != 2:
        raise ValueError("translation_brackets 必须包含一对括号")
    result = [f"[{key}:{value}]" for key, value in document.metadata]
    translations = dict(document.translations)
    for line in document.document.lines:
        if line.source_start is None:
            raise ValueError("当前歌词缺少真实时间戳，无法导出为 LRC")
        minute = int(line.source_start // 60)
        second = line.source_start % 60
        text = line.text
        if include_translations and (translation := translations.get(line.id)):
            text = f"{text}{translation_brackets[0]}{translation}{translation_brackets[1]}"
        result.append(f"[{minute:02d}:{second:05.2f}]{text}")
    return "\n".join(result)
