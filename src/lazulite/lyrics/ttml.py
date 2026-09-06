"""TTML 到自动化歌词输入的只读投影。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as element_tree
from pathlib import Path

from lazulite.models.lyrics import LyricDocument, LyricLine

_CLOCK_TIME = re.compile(r"^(?P<minutes>\d+):(?P<seconds>[0-5]?\d(?:\.\d+)?)$")


class TtmlFormatError(ValueError):
    """输入并非本引擎已明确支持的 TTML 时间和行结构。"""


def _local_name(name: str) -> str:
    return name.rsplit("}", maxsplit=1)[-1]


def _attribute(element: element_tree.Element, name: str) -> str | None:
    for key, value in element.attrib.items():
        if _local_name(key) == name:
            return value
    return None


def _parse_time(value: str, *, field: str) -> float:
    value = value.strip()
    if not value:
        raise TtmlFormatError(f"{field} 不能为空")
    try:
        # 当前素材使用无单位秒数；不要将未声明的帧/tick/sequential 语义猜成秒数。
        if ":" not in value:
            seconds = float(value)
            if seconds < 0:
                raise ValueError
            return seconds
    except ValueError as error:
        raise TtmlFormatError(f"不支持的 {field} 时间表达式：{value!r}") from error

    match = _CLOCK_TIME.fullmatch(value)
    if match is None:
        raise TtmlFormatError(
            f"不支持的 {field} 时间表达式：{value!r}；仅支持秒数或 m:ss.mmm"
        )
    return 60 * float(match.group("minutes")) + float(match.group("seconds"))


def _line_text(element: element_tree.Element) -> str:
    """保留混合文本与 span tail，避免漏掉 span 之间的空格或正文。"""
    return "".join(element.itertext()).strip()


def read_ttml_for_alignment(
    path: str | Path, *, language: str | None = None
) -> LyricDocument:
    """读取 TTML 对齐所需的最小投影，不修改原始 XML 文件。"""
    try:
        root = element_tree.parse(path).getroot()
    except (element_tree.ParseError, OSError) as error:
        raise TtmlFormatError(f"无法读取 TTML：{path}") from error

    document_language = language or _attribute(root, "lang")
    lines: list[LyricLine] = []
    for paragraph in (element for element in root.iter() if _local_name(element.tag) == "p"):
        line_id = _attribute(paragraph, "key") or _attribute(paragraph, "id")
        if not line_id:
            raise TtmlFormatError("每个待对齐 <p> 必须带有稳定的 key 或 id")
        if any(existing.id == line_id for existing in lines):
            raise TtmlFormatError(f"TTML 行 ID 重复：{line_id}")
        text = _line_text(paragraph)
        if not text:
            raise TtmlFormatError(f"TTML 行 {line_id} 缺少正文")
        start_value = _attribute(paragraph, "begin")
        end_value = _attribute(paragraph, "end")
        if start_value is None or end_value is None:
            raise TtmlFormatError(f"TTML 行 {line_id} 必须有 begin 与 end")
        lines.append(
            LyricLine(
                id=line_id,
                text=text,
                source_start=_parse_time(start_value, field=f"行 {line_id} begin"),
                source_end=_parse_time(end_value, field=f"行 {line_id} end"),
                agent_id=_attribute(paragraph, "agent"),
            )
        )
    if not lines:
        raise TtmlFormatError("TTML 中没有可对齐的 <p> 歌词行")
    return LyricDocument(tuple(lines), language=document_language)
