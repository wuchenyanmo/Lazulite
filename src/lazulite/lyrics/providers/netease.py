"""网易云音乐歌词来源适配器。"""

from __future__ import annotations

from itertools import chain
from typing import Any
from urllib.parse import quote
import warnings

import requests
from requests.exceptions import RequestException

from lazulite.lyrics.lrc import LrcDocument, read_lrc, with_translation
from lazulite.lyrics.normalize import clean_text, unique_non_empty_texts
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate

from .base import SearchCandidate

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
}
_SEARCH_LIMIT = 20
_SEARCH_URL = "https://music.163.com/api/search/get?s={name}&type=1&offset=0&limit={limit}"
_LYRIC_URL = "https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"


def parse_netease_artist(artist: dict[str, object]) -> list[str]:
    """提取主名及别名，保持来源原始多入口。"""
    values: list[object | None] = [artist.get("name")]
    for key in ("tns", "alia", "alias"):
        values.extend(artist.get(key) or [])  # type: ignore[arg-type]
    return unique_non_empty_texts(values)


def _song_artists(result: dict[str, object]) -> list[dict[str, object]]:
    artists = result.get("ar") or result.get("artists") or []
    return [artist for artist in artists if isinstance(artist, dict)]  # type: ignore[union-attr]


def _song_album(result: dict[str, object]) -> dict[str, object]:
    album = result.get("al") or result.get("album") or {}
    return album if isinstance(album, dict) else {}


def _song_aliases(result: dict[str, object]) -> list[str]:
    values: list[object | None] = []
    for key in ("alia", "transNames", "tns"):
        values.extend(result.get(key) or [])  # type: ignore[arg-type]
    return unique_non_empty_texts(values)


def build_netease_score_fields(result: dict[str, object]) -> SearchScoreFields:
    album = _song_album(result)
    artists = [parse_netease_artist(artist) for artist in _song_artists(result)]
    duration_value = result.get("dt", result.get("duration"))
    duration = float(duration_value) / 1000.0 if duration_value is not None else None
    return SearchScoreFields(
        duration=duration,
        title_candidates=tuple(unique_non_empty_texts([result.get("name"), *_song_aliases(result)])),
        artist_candidates=tuple(unique_non_empty_texts(chain.from_iterable(artists))),
        album_candidates=tuple(unique_non_empty_texts([album.get("name"), *(album.get("tns") or [])])),  # type: ignore[arg-type]
    )


def candidate_from_netease_result(
    result: dict[str, object],
    *,
    title: str,
    duration: float,
    artist: str | None,
    album: str | None,
) -> SearchCandidate:
    fields = build_netease_score_fields(result)
    artist_names = list(chain.from_iterable(parse_netease_artist(item) for item in _song_artists(result)))
    album_data = _song_album(result)
    candidate_id = clean_text(result.get("id"))
    if not candidate_id:
        raise ValueError("网易云搜索结果缺少歌曲 ID")
    return SearchCandidate(
        source="netease",
        candidate_id=candidate_id,
        title=clean_text(result.get("name")),
        artist=" / ".join(artist_names) if artist_names else None,
        album=clean_text(album_data.get("name")) or None,
        duration=fields.duration,
        match_score=score_search_candidate(
            title=title, duration=duration, fields=fields, artist=artist, album=album
        ),
        raw=dict(result),
    )


def lyrics_from_netease_payload(payload: dict[str, object]) -> LrcDocument | None:
    if payload.get("nolyric") or payload.get("pureMusic"):
        return None
    lrc_payload = payload.get("lrc")
    lyric_text = lrc_payload.get("lyric") if isinstance(lrc_payload, dict) else None
    if not isinstance(lyric_text, str) or not lyric_text.strip():
        return None
    lyrics = read_lrc(lyric_text)
    translation_payload = payload.get("tlyric")
    translation_text = translation_payload.get("lyric") if isinstance(translation_payload, dict) else None
    return with_translation(lyrics, translation_text) if isinstance(translation_text, str) else lyrics


class NeteaseProvider:
    source_name = "netease"
    priority = 100

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

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
        url = _SEARCH_URL.format(name=quote(title), limit=_SEARCH_LIMIT)
        try:
            response = self._session.get(url, headers=_HEADERS, timeout=(5, 7))
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except (RequestException, ValueError) as error:
            raise RuntimeError("网易云搜索请求失败") from error

        if payload.get("code", 200) != 200:
            raise RuntimeError("网易云搜索返回失败状态")
        result = payload.get("result") or {}
        songs = result.get("songs") if isinstance(result, dict) else []
        candidates: list[SearchCandidate] = []
        for song in songs or []:
            if not isinstance(song, dict):
                continue
            try:
                candidates.append(
                    candidate_from_netease_result(
                        song,
                        title=score_title if score_title is not None else title,
                        duration=duration,
                        artist=score_artist if score_artist is not None else artist,
                        album=score_album if score_album is not None else album,
                    )
                )
            except ValueError:
                continue
        return sorted(candidates, key=lambda candidate: candidate.match_score, reverse=True)

    def fetch(self, candidate: SearchCandidate) -> LrcDocument | None:
        if candidate.source != self.source_name:
            raise ValueError("候选来源与网易云适配器不匹配")
        url = _LYRIC_URL.format(song_id=candidate.candidate_id)
        try:
            response = self._session.get(url, headers=_HEADERS, timeout=(5, 7))
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except (RequestException, ValueError) as error:
            warnings.warn(f"网易云歌词请求失败，已跳过该候选: {error}", RuntimeWarning)
            return None
        return lyrics_from_netease_payload(payload)
