"""在线歌词来源的最小协议与通用候选类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from lazulite.lyrics.lrc import LrcDocument


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    source: str
    candidate_id: str
    title: str
    artist: str | None
    album: str | None
    duration: float | None
    match_score: float
    raw: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source or not self.candidate_id:
            raise ValueError("搜索候选必须具有来源和候选 ID")


@dataclass(frozen=True, slots=True)
class FetchedLyricCandidate:
    candidate: SearchCandidate
    lyrics: LrcDocument
    provider_rank: int

    @property
    def lyric_line_count(self) -> int:
        return len(self.lyrics.document.lines)


class LyricProvider(Protocol):
    """在线来源；实现者不需要继承或注册。"""

    source_name: str
    priority: int

    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
        score_title: str | None = None,
        score_artist: str | None = None,
        score_album: str | None = None,
    ) -> list[SearchCandidate]: ...

    def fetch(self, candidate: SearchCandidate) -> LrcDocument | None: ...
