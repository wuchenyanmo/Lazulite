"""LRCLIB 歌词来源适配器。"""

from __future__ import annotations

import re
import warnings
from typing import Any

import requests
from requests.exceptions import RequestException

from lazulite.lyrics.lrc import LrcDocument, read_lrc, read_plain_text
from lazulite.lyrics.normalize import clean_text
from lazulite.lyrics.search import SearchScoreFields, normalize_search_text, score_search_candidate

from .base import SearchCandidate

_HEADERS = {"User-Agent": "Lazulite/0.1"}
_SEARCH_URL = "https://lrclib.net/api/search"
_GET_URL = "https://lrclib.net/api/get/{lyric_id}"
_METADATA = re.compile(r"^\[[A-Za-z]+:.*\]$")


def lyrics_from_lrclib_payload(payload: dict[str, object]) -> LrcDocument | None:
    synced = clean_text(payload.get("syncedLyrics"))
    if synced:
        lyrics = read_lrc(synced)
        if lyrics.document.lines:
            return lyrics
    plain = clean_text(payload.get("plainLyrics"))
    lines = [line for line in plain.splitlines() if line.strip() and not _METADATA.match(line.strip())]
    return LrcDocument(read_plain_text("\n".join(lines))) if lines else None


def candidate_from_lrclib_result(result: dict[str, object], *, title: str, duration: float, artist: str | None, album: str | None) -> SearchCandidate:
    fields = SearchScoreFields(float(result.get("duration") or 0), tuple(filter(None, (clean_text(result.get("trackName")), clean_text(result.get("name"))))), (clean_text(result.get("artistName")),), (clean_text(result.get("albumName")),))
    lyric_id = clean_text(result.get("id"))
    return SearchCandidate("lrclib", lyric_id, clean_text(result.get("trackName") or result.get("name")), clean_text(result.get("artistName")) or None, clean_text(result.get("albumName")) or None, fields.duration, score_search_candidate(title=title, duration=duration, fields=fields, artist=artist, album=album), dict(result))


class LRCLIBProvider:
    source_name = "lrclib"
    priority = 10

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def search(self, title: str, duration: float, artist: str | None = None, album: str | None = None, score_title: str | None = None, score_artist: str | None = None, score_album: str | None = None) -> list[SearchCandidate]:
        variants: list[dict[str, str]] = []
        for track, performer, record in ((title, artist or "", album or ""), (normalize_search_text(title), normalize_search_text(artist), normalize_search_text(album)), (normalize_search_text(title), normalize_search_text(artist), "")):
            item = {"track_name": clean_text(track), "artist_name": clean_text(performer), "album_name": clean_text(record)}
            if item["track_name"] and item not in variants: variants.append(item)
        results: dict[str, dict[str, object]] = {}; last_error: Exception | None = None
        for parameters in variants:
            try:
                response = self._session.get(_SEARCH_URL, params=parameters, headers=_HEADERS, timeout=(5, 10)); response.raise_for_status(); payload = response.json()
            except (RequestException, ValueError) as error:
                last_error = error; continue
            for item in payload or []:
                if isinstance(item, dict) and (item_id := clean_text(item.get("id"))): results.setdefault(item_id, item)
        if not results and last_error:
            warnings.warn(f"LRCLIB 搜索请求失败，已跳过该来源: {last_error}", RuntimeWarning)
        return sorted((candidate_from_lrclib_result(item, title=score_title or title, duration=duration, artist=score_artist if score_artist is not None else artist, album=score_album if score_album is not None else album) for item in results.values()), key=lambda candidate: candidate.match_score, reverse=True)

    def fetch(self, candidate: SearchCandidate) -> LrcDocument | None:
        if candidate.source != self.source_name: raise ValueError("候选来源与 LRCLIB 适配器不匹配")
        if not candidate.candidate_id: return lyrics_from_lrclib_payload(candidate.raw)
        try:
            response = self._session.get(_GET_URL.format(lyric_id=candidate.candidate_id), headers=_HEADERS, timeout=(5, 10)); response.raise_for_status(); return lyrics_from_lrclib_payload(response.json())
        except (RequestException, ValueError) as error:
            warnings.warn(f"LRCLIB 歌词请求失败，已回退搜索载荷: {error}", RuntimeWarning); return lyrics_from_lrclib_payload(candidate.raw)
