"""歌词自动化投影与读音约束。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _validate_optional_interval(
    start: float | None, end: float | None, *, name: str
) -> None:
    for field_name, value in (("start", start), ("end", end)):
        if value is not None and not isfinite(value):
            raise ValueError(f"{name}.{field_name} 必须为有限秒数或 None")
    if start is not None and end is not None and start > end:
        raise ValueError(f"{name} 的开始时间不能晚于结束时间")


@dataclass(frozen=True, slots=True)
class LyricLine:
    """供搜索和对齐使用的一行歌词，不承担完整 TTML 编辑职责。"""

    id: str
    text: str
    source_start: float | None = None
    source_end: float | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("歌词行 id 不能为空")
        _validate_optional_interval(
            self.source_start, self.source_end, name=f"歌词行 {self.id}"
        )


@dataclass(frozen=True, slots=True)
class LyricDocument:
    """保留原文的歌词投影，匹配视图由歌词模块临时生成。"""

    lines: tuple[LyricLine, ...]
    language: str | None = None

    def __post_init__(self) -> None:
        line_ids = [line.id for line in self.lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("LyricDocument 中的歌词行 id 必须唯一")


@dataclass(frozen=True, slots=True)
class ReadingSpan:
    """原文 Unicode code point 范围与确认读音的映射。"""

    line_id: str
    base_start: int
    base_end: int
    reading: str

    def __post_init__(self) -> None:
        if not self.line_id:
            raise ValueError("读音范围必须关联歌词行 id")
        if self.base_start < 0 or self.base_end <= self.base_start:
            raise ValueError("读音范围必须是非空的 Unicode code point 半开区间")
        if not self.reading:
            raise ValueError("读音不能为空")
