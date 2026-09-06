"""QQ 音乐歌词来源适配器。"""

from __future__ import annotations

import base64
from itertools import chain
from typing import Any
import warnings

import requests
from requests.exceptions import RequestException

from lazulite.lyrics.lrc import LrcDocument, read_lrc, read_plain_text, with_translation
from lazulite.lyrics.normalize import clean_text, unique_non_empty_texts
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate

from .base import SearchCandidate

_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://y.qq.com/"}
_SEARCH_URL = "https://c.y.qq.com/soso/fcgi-bin/client_search_cp"
_LYRIC_URL = "https://i.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg"
_MUSICU_URL = "https://u.y.qq.com/cgi-bin/musicu.fcg"


def _artists(result: dict[str, object]) -> list[dict[str, object]]:
    return [item for item in result.get("singer", []) if isinstance(item, dict)]  # type: ignore[union-attr]


def _album(result: dict[str, object]) -> dict[str, object]:
    value = result.get("album")
    return value if isinstance(value, dict) else {"name": result.get("albumname")}


def _decode(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("[") or "\n[" in text:
        return text
    try:
        return base64.b64decode(text).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return text


def lyrics_from_qq_payload(payload: dict[str, object]) -> LrcDocument | None:
    if int(payload.get("retcode", payload.get("code", 0))) != 0:
        return None
    text = _decode(payload.get("lyric"))
    if not text:
        return None
    lyrics = read_lrc(text)
    if not lyrics.document.lines:
        lyrics = LrcDocument(read_plain_text(text))
    translation = _decode(payload.get("trans"))
    return with_translation(lyrics, translation) if translation else lyrics


def candidate_from_qq_result(result: dict[str, object], *, title: str, duration: float, artist: str | None, album: str | None) -> SearchCandidate:
    album_data = _album(result)
    names = unique_non_empty_texts([result.get("name"), result.get("songname"), result.get("title"), result.get("subtitle")])
    artists = [unique_non_empty_texts([item.get("name"), item.get("title")]) for item in _artists(result)]
    fields = SearchScoreFields(float(result.get("interval") or 0), tuple(names), tuple(unique_non_empty_texts(chain.from_iterable(artists))), tuple(unique_non_empty_texts([album_data.get("name"), album_data.get("title")])))
    candidate_id = clean_text(result.get("mid") or result.get("songmid"))
    if not candidate_id:
        raise ValueError("QQ 音乐搜索结果缺少 songmid")
    return SearchCandidate("qqmusic", candidate_id, names[0] if names else "", " / ".join(chain.from_iterable(artists)) or None, clean_text(album_data.get("name")) or None, fields.duration, score_search_candidate(title=title, duration=duration, fields=fields, artist=artist, album=album), dict(result))


class QQMusicProvider:
    source_name = "qqmusic"
    priority = 100

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def search(self, title: str, duration: float, artist: str | None = None, album: str | None = None, score_title: str | None = None, score_artist: str | None = None, score_album: str | None = None) -> list[SearchCandidate]:
        params = {"format": "json", "platform": "yqq.json", "w": " ".join(part for part in (title, artist) if part), "p": 1, "n": 20}
        try:
            payload: dict[str, Any] = self._session.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=(5, 7)).json()
        except (RequestException, ValueError) as error:
            warnings.warn(f"QQ 音乐搜索请求失败，已跳过该来源: {error}", RuntimeWarning); return []
        songs = ((payload.get("data") or {}).get("song") or {}).get("list", [])
        candidates = [candidate_from_qq_result(song, title=score_title or title, duration=duration, artist=score_artist if score_artist is not None else artist, album=score_album if score_album is not None else album) for song in songs if isinstance(song, dict) and (song.get("mid") or song.get("songmid"))]
        return sorted(candidates, key=lambda item: item.match_score, reverse=True)

    def fetch(self, candidate: SearchCandidate) -> LrcDocument | None:
        if candidate.source != self.source_name: raise ValueError("候选来源与 QQ 音乐适配器不匹配")
        params = {"songmid": candidate.candidate_id, "g_tk": 5381, "format": "json", "nobase64": 1}
        try:
            response = self._session.get(_LYRIC_URL, params=params, headers=_HEADERS, timeout=(5, 15)); response.raise_for_status(); payload = response.json()
        except (RequestException, ValueError) as error:
            warnings.warn(f"QQ 音乐歌词请求失败，已跳过该候选: {error}", RuntimeWarning); return None
        return lyrics_from_qq_payload(payload)
