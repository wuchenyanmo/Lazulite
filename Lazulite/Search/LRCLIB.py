from __future__ import annotations

import re
import warnings

import numpy as np
import requests
from requests.exceptions import RequestException

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import combined_fuzzy_score, normalize_search_text
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate
from Lazulite.TextNormalize import clean_text

LRCLIB_HEADERS = {
    "User-Agent": "LyricPlusScript/0.1 (+https://github.com/)",
}
LRCLIB_SEARCH_API = "https://lrclib.net/api/search"
LRCLIB_GET_BY_ID_API = "https://lrclib.net/api/get/{lyric_id}"
RE_LRCLIB_METADATA_LINE = re.compile(r"^\[[a-zA-Z]+:.*\]$")
SEARCH_LIMIT = 20


def match_lrclib_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
    name_weight: float = 0.7,
    album_weight: float = 0.2,
    artist_weight: float = 0.1,
    full_match_weight: float = 0.4,
    duration_threshold: float = 15.0,
) -> float:
    song_duration = float(result.get("duration") or 0.0)
    if np.abs(song_duration - duration) > duration_threshold:
        return 0.0

    result_names = [
        str(result.get("trackName") or "").strip(),
        str(result.get("name") or "").strip(),
    ]
    result_names = [item for item in result_names if item]
    score = {
        "name": max(combined_fuzzy_score(name, item, full_match_weight=full_match_weight) for item in result_names)
        if result_names else 0.0,
    }

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        artist_name = str(result.get("artistName") or "").strip()
        score["artist"] = combined_fuzzy_score(artist, artist_name, full_match_weight=full_match_weight) if artist_name else 0.0

    if album is None:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        album_name = str(result.get("albumName") or "").strip()
        score["album"] = combined_fuzzy_score(album, album_name, full_match_weight=full_match_weight) if album_name else 0.0

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    value = (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight
    return float(value)


def _parse_lrclib_lyric(payload: dict) -> LyricLineStamp | None:
    synced_lyrics = str(payload.get("syncedLyrics") or "").strip()
    if synced_lyrics:
        lyric = LyricLineStamp(synced_lyrics)
        if lyric.lyric_lines:
            return lyric

    plain_lyrics = str(payload.get("plainLyrics") or "").strip()
    if not plain_lyrics:
        return None
    cleaned_lines = [
        line for line in plain_lyrics.splitlines()
        if line.strip() and not RE_LRCLIB_METADATA_LINE.match(line.strip())
    ]
    if not cleaned_lines:
        return None
    return LyricLineStamp.from_plain_text("\n".join(cleaned_lines))
class LRCLIBProvider(OnlineLyricProvider):
    source_name = "lrclib"

    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
        score_title: str | None = None,
        score_artist: str | None = None,
        score_album: str | None = None,
        limit: int = SEARCH_LIMIT,
    ) -> list[SearchCandidate]:
        score_title = score_title if score_title is not None else title
        score_artist = score_artist if score_artist is not None else artist
        score_album = score_album if score_album is not None else album
        query_variants: list[dict[str, str]] = []
        normalized_title = normalize_search_text(title)
        normalized_artist = normalize_search_text(artist)
        normalized_album = normalize_search_text(album)
        for track_name, artist_name, album_name in [
            (title, artist or "", album or ""),
            (normalized_title, normalized_artist or artist or "", normalized_album or album or ""),
            (normalized_title, normalized_artist or artist or "", ""),
            (title, artist or "", ""),
        ]:
            params = {
                "track_name": str(track_name or "").strip(),
                "artist_name": clean_text(artist_name),
                "album_name": clean_text(album_name),
            }
            if not params["track_name"]:
                continue
            if params not in query_variants:
                query_variants.append(params)

        results_by_id: dict[str, dict] = {}
        last_error: RequestException | None = None
        for params in query_variants:
            try:
                response = requests.get(LRCLIB_SEARCH_API, params=params, headers=LRCLIB_HEADERS, timeout=(5, 10))
                response.raise_for_status()
                payload = response.json()
            except RequestException as exc:
                last_error = exc
                continue
            for item in list(payload or []):
                lyric_id = str(item.get("id") or "")
                if not lyric_id or lyric_id in results_by_id:
                    continue
                results_by_id[lyric_id] = item

        if not results_by_id and last_error is not None:
            warnings.warn(f"LRCLIB 搜索请求失败，已跳过该来源: {last_error}", RuntimeWarning)
            return []

        results = list(results_by_id.values())[:max(1, int(limit))]
        candidates: list[SearchCandidate] = []
        for item in results:
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("id") or ""),
                    title=str(item.get("trackName") or item.get("name") or "").strip(),
                    artist=str(item.get("artistName") or "").strip() or None,
                    album=str(item.get("albumName") or "").strip() or None,
                    duration=float(item.get("duration") or 0.0),
                    match_score=match_lrclib_search_result(score_title, duration, item, score_artist, score_album),
                    raw=item,
                )
            )
        candidates.sort(key=lambda item: item.match_score, reverse=True)
        return candidates

    def fetch_lyric(self, candidate: SearchCandidate) -> LyricLineStamp | None:
        lyric_id = candidate.candidate_id
        if not lyric_id:
            return _parse_lrclib_lyric(candidate.raw)
        url = LRCLIB_GET_BY_ID_API.format(lyric_id=lyric_id)
        try:
            response = requests.get(url, headers=LRCLIB_HEADERS, timeout=(5, 10))
            response.raise_for_status()
            payload = response.json()
        except RequestException as exc:
            warnings.warn(f"LRCLIB 歌词请求失败，已跳过该候选: {exc}", RuntimeWarning)
            return _parse_lrclib_lyric(candidate.raw)
        return _parse_lrclib_lyric(payload)

    def fetch_lyric_by_id(self, lyric_id: str | int) -> LyricLineStamp | None:
        candidate = SearchCandidate(
            source=self.source_name,
            candidate_id=str(lyric_id),
            title="",
            artist=None,
            album=None,
            duration=None,
            match_score=0.0,
        )
        return self.fetch_lyric(candidate)


def search_lrclib_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
) -> list[SearchCandidate]:
    return LRCLIBProvider().search(name, duration, artist, album)


def get_lrclib_lyric(lyric_id: str | int) -> LyricLineStamp | None:
    return LRCLIBProvider().fetch_lyric_by_id(lyric_id)
