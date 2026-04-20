from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from Lazulite.Lyric import LyricLineStamp


@dataclass(slots=True)
class SearchCandidate:
    source: str
    candidate_id: str
    title: str
    artist: str | None
    album: str | None
    duration: float | None
    match_score: float
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "candidate_id": self.candidate_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "duration": self.duration,
            "match_score": self.match_score,
            "raw": self.raw,
        }


class OnlineLyricProvider(ABC):
    source_name: str
    priority: int = 100

    @abstractmethod
    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
        score_title: str | None = None,
        score_artist: str | None = None,
        score_album: str | None = None,
    ) -> list[SearchCandidate]:
        """
        根据歌曲元数据搜索候选。
        """

    @abstractmethod
    def fetch_lyric(self, candidate: SearchCandidate) -> LyricLineStamp | None:
        """
        根据候选获取歌词。
        """
