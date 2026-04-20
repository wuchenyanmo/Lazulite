from __future__ import annotations

import base64
import warnings
from itertools import chain

import requests
from requests.exceptions import RequestException

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import SearchScoreFields, score_search_candidate
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate
from Lazulite.TextNormalize import clean_text, unique_non_empty_texts

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
        value = clean_text(result.get(key))
        if value:
            aliases.append(value)
    return aliases


def parse_qq_artist_dict(artist_dict: dict) -> list[str]:
    return unique_non_empty_texts([artist_dict.get("name"), artist_dict.get("title")])


def build_qq_score_fields(result: dict) -> SearchScoreFields:
    album_info = _song_album(result)
    result_artists = [parse_qq_artist_dict(item) for item in _song_artists(result)]
    return SearchScoreFields(
        duration=float(result.get("interval") or 0.0),
        title_candidates=unique_non_empty_texts([result.get("name"), *_song_aliases(result)]),
        artist_candidates=unique_non_empty_texts(chain(*result_artists)),
        album_candidates=unique_non_empty_texts([
            album_info.get("name"),
            album_info.get("title"),
            album_info.get("subtitle"),
        ]),
    )


def match_qq_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
) -> float:
    return score_search_candidate(
        title=name,
        duration=duration,
        fields=build_qq_score_fields(result),
        artist=artist,
        album=album,
    )


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
    priority = 100

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
            score_fields = build_qq_score_fields(item)
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("mid") or ""),
                    title=str(item.get("name") or item.get("title") or "").strip(),
                    artist=" / ".join(artist_names) if artist_names else None,
                    album=str(album_info.get("name") or album_info.get("title") or "").strip() or None,
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
