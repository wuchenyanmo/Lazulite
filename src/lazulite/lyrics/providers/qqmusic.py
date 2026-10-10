"""QQ 音乐歌词来源适配器。"""

from __future__ import annotations

import base64
from itertools import chain
import warnings

import requests
from requests.exceptions import RequestException

from lazulite.lyrics.lrc import LrcDocument, read_lrc, read_plain_text, with_translation
from lazulite.lyrics.normalize import clean_text, unique_non_empty_texts
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate

from .base import SearchCandidate

_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://y.qq.com/"}
_SEARCH_URL = "https://c.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg"
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


def candidate_from_qq_result(
    result: dict[str, object],
    *,
    title: str,
    duration: float,
    artist: str | None,
    album: str | None,
) -> SearchCandidate:
    album_data = _album(result)
    names = unique_non_empty_texts(
        [
            result.get("name"),
            result.get("songname"),
            result.get("title"),
            result.get("subtitle"),
        ]
    )
    artists = [
        unique_non_empty_texts([item.get("name"), item.get("title")])
        for item in _artists(result)
    ]
    fields = SearchScoreFields(
        float(result.get("interval") or 0),
        tuple(names),
        tuple(unique_non_empty_texts(chain.from_iterable(artists))),
        tuple(
            unique_non_empty_texts(
                [
                    album_data.get("name"),
                    album_data.get("title"),
                    album_data.get("subtitle"),
                ]
            )
        ),
    )
    candidate_id = clean_text(result.get("mid") or result.get("songmid"))
    if not candidate_id:
        raise ValueError("QQ 音乐搜索结果缺少 songmid")
    return SearchCandidate(
        "qqmusic",
        candidate_id,
        names[0] if names else "",
        " / ".join(chain.from_iterable(artists)) or None,
        clean_text(album_data.get("name")) or None,
        fields.duration,
        score_search_candidate(
            title=title, duration=duration, fields=fields, artist=artist, album=album
        ),
        dict(result),
    )


class QQMusicProvider:
    source_name = "qqmusic"
    priority = 100

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._details = {}

    def _request(self, url, *, params=None, body=None):
        last_error = None
        for timeout in ((5, 7), (5, 15)):
            try:
                if body is None:
                    response = self._session.get(
                        url, params=params, headers=_HEADERS, timeout=timeout
                    )
                else:
                    response = self._session.post(
                        url, json=body, headers=_HEADERS, timeout=timeout
                    )
                response.raise_for_status()
                return response.json()
            except (RequestException, ValueError) as error:
                last_error = error
        raise RuntimeError(f"QQ 音乐请求失败: {url}") from last_error

    def _musicu(self, module, method, parameters):
        payload = self._request(
            _MUSICU_URL,
            body={
                "req_0": {"module": module, "method": method, "param": parameters},
                "loginUin": "0",
                "comm": {"uin": "0", "format": "json", "ct": 24, "cv": 0},
            },
        )
        response = payload.get("req_0") or {}
        if payload.get("code") != 0 or response.get("code") != 0:
            raise RuntimeError("QQ musicu 返回失败状态")
        return response.get("data") or {}

    def _detail(self, song):
        key = str(song.get("id") or song.get("mid") or "")
        if key not in self._details:
            parameters = (
                {"song_id": int(song["id"])}
                if song.get("id")
                else {"song_mid": song["mid"]}
            )
            result = self._musicu(
                "music.pf_song_detail_svr", "get_song_detail_yqq", parameters
            )
            track = result.get("track_info") or {}
            if not track.get("mid") or not track.get("interval"):
                raise ValueError("QQ 详情缺少歌曲 ID 或时长，无法评分")
            self._details[key] = track
        return self._details[key]

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
        payload = self._request(
            _SEARCH_URL,
            params={"format": "json", "key": " ".join(v for v in (title, artist) if v)},
        )
        if payload.get("code") != 0:
            raise RuntimeError("QQ 搜索失败")
        songs = ((payload.get("data") or {}).get("song") or {}).get("itemlist") or []
        candidates = []
        errors = []
        for song in songs[:20]:
            try:
                detail = self._detail(song)
                candidates.append(
                    candidate_from_qq_result(
                        detail,
                        title=score_title or title,
                        duration=duration,
                        artist=score_artist if score_artist is not None else artist,
                        album=score_album if score_album is not None else album,
                    )
                )
            except (RuntimeError, ValueError) as error:
                errors.append(error)
        if songs and not candidates and errors:
            raise RuntimeError("QQ 搜索结果详情全部获取失败") from errors[0]
        for error in errors:
            warnings.warn(str(error), RuntimeWarning)
        return sorted(
            candidates, key=lambda item: (-item.match_score, item.candidate_id)
        )

    def fetch(self, candidate: SearchCandidate) -> LrcDocument | None:
        if candidate.source != self.source_name:
            raise ValueError("候选来源与 QQ 音乐适配器不匹配")
        failure = None
        try:
            payload = self._musicu(
                "music.musichallSong.PlayLyricInfo",
                "GetPlayLyricInfo",
                {
                    "songMID": candidate.candidate_id,
                    "songID": int(
                        candidate.raw.get("id") or candidate.raw.get("songid") or 0
                    ),
                    "crypt": 0,
                    "type": 1,
                    "lrc_t": 0,
                    "qrc": 0,
                    "qrc_t": 0,
                    "trans": 1,
                    "trans_t": 0,
                    "roma_t": 0,
                },
            )
            # Request verified unencrypted line lyrics, not Studio's encrypted QRC representation.
            if payload.get("crypt") not in (None, 0) or payload.get("qrc") not in (
                None,
                0,
            ):
                raise ValueError("QQ 返回了非逐行歌词格式")
            lyrics = lyrics_from_qq_payload(payload)
            if lyrics is not None and lyrics.document.lines:
                return lyrics
        except (RuntimeError, ValueError) as error:
            failure = error
        payload = self._request(
            _LYRIC_URL,
            params={
                "songmid": candidate.candidate_id,
                "g_tk": 5381,
                "format": "json",
                "nobase64": 1,
            },
        )
        if int(payload.get("retcode", payload.get("code", 0))) != 0:
            raise RuntimeError("QQ 歌词主入口及回退入口失败") from failure
        return lyrics_from_qq_payload(payload)
