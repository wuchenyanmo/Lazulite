from __future__ import annotations

import base64
import warnings
from itertools import chain

import numpy as np
import requests
from requests.exceptions import RequestException

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import combined_fuzzy_score
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate

QQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Referer": "https://y.qq.com/",
}
QQ_SEARCH_API = "https://u.y.qq.com/cgi-bin/musicu.fcg"
QQ_LYRIC_API = "https://i.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg"
SEARCH_LIMIT = 20


def _song_artists(result: dict) -> list[dict]:
    return list(result.get("singer") or [])


def _song_album(result: dict) -> dict:
    return dict(result.get("album") or {})


def _song_aliases(result: dict) -> list[str]:
    aliases: list[str] = []
    for key in ("subtitle", "title"):
        value = str(result.get(key) or "").strip()
        if value:
            aliases.append(value)
    return aliases


def parse_qq_artist_dict(artist_dict: dict) -> list[str]:
    artist_list = [str(artist_dict.get("name") or "").strip(), str(artist_dict.get("title") or "").strip()]
    return [item for item in set(artist_list) if item]


def match_qq_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
    name_weight: float = 0.7,
    album_weight: float = 0.2,
    artist_weight: float = 0.1,
    full_match_weight: float = 0.2,
    duration_threshold: float = 15.0,
) -> float:
    song_duration = float(result.get("interval") or 0.0)
    if np.abs(song_duration - duration) > duration_threshold:
        return 0.0

    result_names = [str(result.get("name") or "").strip(), *_song_aliases(result)]
    result_names = [item for item in result_names if item]
    score = {
        "name": max(combined_fuzzy_score(name, item, full_match_weight=full_match_weight) for item in result_names)
        if result_names else 0.0,
    }

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        result_artists = [parse_qq_artist_dict(item) for item in _song_artists(result)]
        result_artists = list(chain(*result_artists))
        score["artist"] = (
            max(combined_fuzzy_score(artist, item, full_match_weight=full_match_weight) for item in result_artists)
            if result_artists else 0.0
        )

    album_info = _song_album(result)
    if album is None or not album_info:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        result_albums = [
            str(album_info.get("name") or "").strip(),
            str(album_info.get("title") or "").strip(),
            str(album_info.get("subtitle") or "").strip(),
        ]
        result_albums = [item for item in result_albums if item]
        score["album"] = (
            max(combined_fuzzy_score(album, item, full_match_weight=full_match_weight) for item in result_albums)
            if result_albums else 0.0
        )

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    value = (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight
    return float(value)


def _decode_qq_lyric_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).strip()
    if not normalized:
        return ""
    if normalized.startswith("[") or "\n[" in normalized:
        return normalized
    try:
        decoded = base64.b64decode(normalized)
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return normalized


def _parse_qq_lyric_text(lyric_text: str) -> LyricLineStamp | None:
    normalized = lyric_text.strip()
    if not normalized:
        return None
    lyric = LyricLineStamp(normalized)
    if lyric.lyric_lines:
        return lyric
    return LyricLineStamp.from_plain_text(normalized)


class QQMusicProvider(OnlineLyricProvider):
    source_name = "qqmusic"

    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
        limit: int = SEARCH_LIMIT,
    ) -> list[SearchCandidate]:
        payload = {
            "comm": {"ct": "19", "cv": "1859", "uin": "0"},
            "req": {
                "method": "DoSearchForQQMusicDesktop",
                "module": "music.search.SearchCgiService",
                "param": {
                    "grp": 1,
                    "num_per_page": max(1, int(limit)),
                    "page_num": 1,
                    "query": " ".join(part for part in [title, artist] if part),
                    "search_type": 0,
                },
            },
        }
        try:
            response = requests.post(QQ_SEARCH_API, json=payload, headers=QQ_HEADERS, timeout=(5, 7))
            response.raise_for_status()
            data = response.json()
        except RequestException as exc:
            warnings.warn(f"QQ 音乐搜索请求失败，已跳过该来源: {exc}", RuntimeWarning)
            return []

        songs = (((data.get("req") or {}).get("data") or {}).get("body") or {}).get("song", {}).get("list", [])
        candidates: list[SearchCandidate] = []
        for item in songs:
            artist_names = list(chain(*[parse_qq_artist_dict(artist_item) for artist_item in _song_artists(item)]))
            album_info = _song_album(item)
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("mid") or ""),
                    title=str(item.get("name") or item.get("title") or "").strip(),
                    artist=" / ".join(artist_names) if artist_names else None,
                    album=str(album_info.get("name") or album_info.get("title") or "").strip() or None,
                    duration=float(item.get("interval") or 0.0),
                    match_score=match_qq_search_result(title, duration, item, artist, album),
                    raw=item,
                )
            )
        candidates.sort(key=lambda item: item.match_score, reverse=True)
        return candidates

    def fetch_lyric(self, candidate: SearchCandidate) -> LyricLineStamp | None:
        params = {
            "songmid": candidate.candidate_id,
            "g_tk": 5381,
            "format": "json",
            "inCharset": "utf8",
            "outCharset": "utf-8",
            "notice": 0,
            "platform": "yqq.json",
            "needNewCode": 0,
            "nobase64": 1,
        }
        payload = None
        last_error: RequestException | None = None
        for timeout in ((5, 7), (5, 15)):
            try:
                response = requests.get(QQ_LYRIC_API, params=params, headers=QQ_HEADERS, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except RequestException as exc:
                last_error = exc
        if payload is None:
            warnings.warn(f"QQ 音乐歌词请求失败，已跳过该候选: {last_error}", RuntimeWarning)
            return None

        retcode = payload.get("retcode")
        if retcode is None:
            retcode = payload.get("code", -1)
        if int(retcode) != 0:
            return None

        lyric_text = _decode_qq_lyric_text(payload.get("lyric"))
        if not lyric_text:
            return None

        lyric = _parse_qq_lyric_text(lyric_text)
        if lyric is None:
            return None
        translation_text = _decode_qq_lyric_text(payload.get("trans"))
        if translation_text:
            lyric.load_translation(translation_text)
        return lyric

    def fetch_lyric_by_song_mid(self, song_mid: str) -> LyricLineStamp | None:
        candidate = SearchCandidate(
            source=self.source_name,
            candidate_id=song_mid,
            title="",
            artist=None,
            album=None,
            duration=None,
            match_score=0.0,
        )
        return self.fetch_lyric(candidate)


def search_qq_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
) -> list[SearchCandidate]:
    return QQMusicProvider().search(name, duration, artist, album)


def get_qq_lyric(song_mid: str) -> LyricLineStamp | None:
    return QQMusicProvider().fetch_lyric_by_song_mid(song_mid)
