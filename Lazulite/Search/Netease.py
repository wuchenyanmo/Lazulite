from __future__ import annotations

from itertools import chain
from urllib.parse import quote
import warnings

import numpy as np
import requests
from requests.exceptions import RequestException

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import combined_fuzzy_score
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate

HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
}
SEARCH_LIMIT = 20
# 之前这里用过第三方 `apis.netstart.cn`，注释里也明确写过它“信息更全”。
# 这次重新核对接口后，官方搜索接口已经能稳定返回当前打分所需的字段：
# `id` / `dt` / `ar` / `al` / `alia` / `tns`。
# 因此这里优先切回官方接口，减少对第三方服务的依赖面。
SEARCH_163_API = "https://music.163.com/api/search/get?s={name}&type=1&offset=0&limit={limit}"
LYRIC_163_API = "https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"


def parse_163_artist_dict(artist_dict: dict) -> list[str]:
    """
    解析网易云歌手对象，返回去重后的候选名字列表。
    """
    artist_list = [str(artist_dict.get("name") or "").strip()]
    for key in ("tns", "alia", "alias"):
        values = artist_dict.get(key) or []
        artist_list.extend(str(item).strip() for item in values if str(item).strip())
    return [item for item in set(artist_list) if item]


def _song_artists(result: dict) -> list[dict]:
    artists = result.get("ar")
    if artists:
        return artists
    return result.get("artists") or []


def _song_album(result: dict) -> dict:
    album = result.get("al")
    if album:
        return album
    return result.get("album") or {}


def _song_aliases(result: dict) -> list[str]:
    aliases: list[str] = []
    for key in ("alia", "transNames", "tns"):
        values = result.get(key) or []
        aliases.extend(str(item).strip() for item in values if str(item).strip())
    return aliases


def match_163_search_result(
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
    """
    根据歌曲名、歌手和专辑对网易搜索结果打分。
    """
    song_duration = float(result.get("dt", result.get("duration", 0.0))) / 1000.0
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
        result_artists = [parse_163_artist_dict(item) for item in _song_artists(result)]
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
        result_albums = [str(album_info.get("name") or "").strip(), *(album_info.get("tns") or [])]
        result_albums = [str(item).strip() for item in result_albums if str(item).strip()]
        score["album"] = (
            max(combined_fuzzy_score(album, item, full_match_weight=full_match_weight) for item in result_albums)
            if result_albums else 0.0
        )

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    value = (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight
    return float(value)


class NeteaseProvider(OnlineLyricProvider):
    source_name = "netease"

    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
    ) -> list[SearchCandidate]:
        url = SEARCH_163_API.format(name=quote(title), limit=SEARCH_LIMIT)
        try:
            response = requests.get(url, headers=HEADER, timeout=(5, 7))
            response.raise_for_status()
            payload = response.json()
        except RequestException as exc:
            warnings.warn(f"网易云搜索请求失败，已跳过该来源: {exc}", RuntimeWarning)
            return []

        songs = payload.get("result", {}).get("songs", [])
        candidates: list[SearchCandidate] = []
        for item in songs:
            artist_names = list(chain(*[parse_163_artist_dict(artist_item) for artist_item in _song_artists(item)]))
            album_info = _song_album(item)
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("id")),
                    title=str(item.get("name") or "").strip(),
                    artist=" / ".join(artist_names) if artist_names else None,
                    album=str(album_info.get("name") or "").strip() or None,
                    duration=float(item.get("dt", item.get("duration", 0.0))) / 1000.0,
                    match_score=match_163_search_result(title, duration, item, artist, album),
                    raw=item,
                )
            )
        candidates.sort(key=lambda item: item.match_score, reverse=True)
        return candidates

    def fetch_lyric(self, candidate: SearchCandidate) -> LyricLineStamp | None:
        return self.fetch_lyric_by_song_id(candidate.candidate_id)

    def fetch_lyric_by_song_id(self, song_id: str | int) -> LyricLineStamp | None:
        url = LYRIC_163_API.format(song_id=song_id)
        try:
            response = requests.get(url, headers=HEADER, timeout=(5, 7))
            response.raise_for_status()
            payload = response.json()
        except RequestException as exc:
            warnings.warn(f"网易云歌词请求失败，已跳过该候选: {exc}", RuntimeWarning)
            return None

        if "lrc" not in payload:
            return None
        if payload.get("nolyric") or payload.get("pureMusic"):
            return None

        lyric_text = (payload.get("lrc") or {}).get("lyric")
        if not lyric_text:
            return None

        lyric = LyricLineStamp(lyric_text)
        translation_text = (payload.get("tlyric") or {}).get("lyric")
        if translation_text:
            lyric.load_translation(translation_text)
        return lyric


def search_163_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
) -> list[SearchCandidate]:
    return NeteaseProvider().search(name, duration, artist, album)


def get_163_lyric(song_id: str | int) -> LyricLineStamp | None:
    return NeteaseProvider().fetch_lyric_by_song_id(song_id)
