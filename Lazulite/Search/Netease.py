from __future__ import annotations

from itertools import chain
from urllib.parse import quote
import warnings

import requests
from requests.exceptions import RequestException

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import SearchScoreFields, score_search_candidate
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate
from Lazulite.TextNormalize import clean_text, unique_non_empty_texts

HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
}
SEARCH_LIMIT = 20
SEARCH_163_API = "https://music.163.com/api/search/get?s={name}&type=1&offset=0&limit={limit}"
LYRIC_163_API = "https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"


def parse_163_artist_dict(artist_dict: dict) -> list[str]:
    """
    解析网易云歌手对象，返回去重后的候选名字列表。
    """
    artist_list = [clean_text(artist_dict.get("name"))]
    for key in ("tns", "alia", "alias"):
        values = artist_dict.get(key) or []
        artist_list.extend(values)
    return unique_non_empty_texts(artist_list)


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
        aliases.extend(values)
    return unique_non_empty_texts(aliases)


def build_163_score_fields(result: dict) -> SearchScoreFields:
    album_info = _song_album(result)
    result_artists = [parse_163_artist_dict(item) for item in _song_artists(result)]
    return SearchScoreFields(
        duration=float(result.get("dt", result.get("duration", 0.0))) / 1000.0,
        title_candidates=unique_non_empty_texts([result.get("name"), *_song_aliases(result)]),
        artist_candidates=unique_non_empty_texts(chain(*result_artists)),
        album_candidates=unique_non_empty_texts([album_info.get("name"), *(album_info.get("tns") or [])]),
    )


def match_163_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
) -> float:
    return score_search_candidate(
        title=name,
        duration=duration,
        fields=build_163_score_fields(result),
        artist=artist,
        album=album,
    )


class NeteaseProvider(OnlineLyricProvider):
    source_name = "netease"

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
        score_title = score_title if score_title is not None else title
        score_artist = score_artist if score_artist is not None else artist
        score_album = score_album if score_album is not None else album
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
            score_fields = build_163_score_fields(item)
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("id")),
                    title=str(item.get("name") or "").strip(),
                    artist=" / ".join(artist_names) if artist_names else None,
                    album=str(album_info.get("name") or "").strip() or None,
                    duration=score_fields.duration,
                    match_score=score_search_candidate(
                        title=score_title,
                        duration=duration,
                        fields=score_fields,
                        artist=score_artist,
                        album=score_album,
                    ),
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
