from __future__ import annotations

import base64
import warnings

import requests
from requests.exceptions import RequestException, SSLError
from urllib3.exceptions import InsecureRequestWarning

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import SearchScoreFields, score_search_candidate
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate
from Lazulite.TextNormalize import KUGOU_ARTIST_SEPARATORS, split_text

KUGOU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.kugou.com/",
}
KUGOU_SEARCH_API = "https://mobilecdn.kugou.com/api/v3/search/song"
KUGOU_LYRIC_SEARCH_API = "https://lyrics.kugou.com/search"
KUGOU_LYRIC_DOWNLOAD_API = "https://lyrics.kugou.com/download"


def _safe_json_get(url: str, params: dict, timeout: tuple[int, int] = (5, 7)) -> dict:
    try:
        response = requests.get(url, params=params, headers=KUGOU_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except SSLError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            try:
                response = requests.get(url, params=params, headers=KUGOU_HEADERS, timeout=timeout, verify=False)
                response.raise_for_status()
                return response.json()
            except RequestException as exc:
                warnings.warn(f"酷狗请求失败，已跳过当前请求: {exc}", RuntimeWarning)
                return {}
    except RequestException as exc:
        warnings.warn(f"酷狗请求失败，已跳过当前请求: {exc}", RuntimeWarning)
        return {}


def _split_kugou_artists(value: str | None) -> list[str]:
    return split_text(value, KUGOU_ARTIST_SEPARATORS)


def _build_kugou_keyword(candidate: SearchCandidate) -> str:
    if candidate.artist and candidate.title:
        return f"{candidate.artist} - {candidate.title}"
    if candidate.title:
        return candidate.title
    return candidate.candidate_id


def _kugou_candidate_names(result: dict) -> list[str]:
    values = [
        result.get("songname"),
        result.get("songname_original"),
        result.get("filename"),
        result.get("othername"),
        result.get("othername_original"),
        result.get("remark"),
    ]
    return [str(item).strip() for item in values if item]


def build_kugou_score_fields(result: dict) -> SearchScoreFields:
    return SearchScoreFields(
        duration=float(result.get("duration") or 0.0),
        title_candidates=_kugou_candidate_names(result),
        artist_candidates=_split_kugou_artists(result.get("singername")),
        album_candidates=[str(result.get("album_name") or "").strip()] if str(result.get("album_name") or "").strip() else [],
    )


def match_kugou_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
) -> float:
    return score_search_candidate(
        title=name,
        duration=duration,
        fields=build_kugou_score_fields(result),
        artist=artist,
        album=album,
    )


class KugouProvider(OnlineLyricProvider):
    source_name = "kugou"
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
        pagesize: int = 20,
    ) -> list[SearchCandidate]:
        score_title = score_title if score_title is not None else title
        score_artist = score_artist if score_artist is not None else artist
        score_album = score_album if score_album is not None else album
        params = {
            "format": "json",
            "keyword": title,
            "page": 1,
            "pagesize": max(1, int(pagesize)),
            "showtype": 1,
        }
        payload = _safe_json_get(KUGOU_SEARCH_API, params=params)
        songs = (payload.get("data") or {}).get("info", [])
        candidates: list[SearchCandidate] = []
        for item in songs:
            artist_name = " / ".join(_split_kugou_artists(item.get("singername"))) or None
            score_fields = build_kugou_score_fields(item)
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("hash") or ""),
                    title=str(item.get("songname") or item.get("songname_original") or "").strip(),
                    artist=artist_name,
                    album=str(item.get("album_name") or "").strip() or None,
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

    def search_lyric_candidates(
        self,
        keyword: str,
        duration_ms: int | None = None,
        hash_value: str | None = None,
    ) -> list[dict]:
        params = {
            "ver": 1,
            "man": "yes",
            "client": "pc",
            "keyword": keyword,
            "duration": duration_ms or 0,
            "hash": hash_value or "",
        }
        payload = _safe_json_get(KUGOU_LYRIC_SEARCH_API, params=params)
        return list(payload.get("candidates") or [])

    def download_lyric(self, lyric_id: str | int, accesskey: str, fmt: str = "lrc") -> str | None:
        params = {
            "ver": 1,
            "client": "pc",
            "id": lyric_id,
            "accesskey": accesskey,
            "fmt": fmt,
            "charset": "utf8",
        }
        payload = _safe_json_get(KUGOU_LYRIC_DOWNLOAD_API, params=params)
        encoded = payload.get("content")
        if not encoded:
            return None
        try:
            raw = base64.b64decode(encoded)
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    def fetch_lyric(self, candidate: SearchCandidate) -> LyricLineStamp | None:
        duration_ms = int(round(float(candidate.duration or 0.0) * 1000))
        lyric_keyword = _build_kugou_keyword(candidate)
        lyric_candidates = self.search_lyric_candidates(
            keyword=lyric_keyword,
            duration_ms=duration_ms,
            hash_value=candidate.candidate_id,
        )
        for lyric_candidate in lyric_candidates:
            lyric_text = self.download_lyric(
                lyric_id=lyric_candidate.get("id"),
                accesskey=str(lyric_candidate.get("accesskey") or ""),
                fmt="lrc",
            )
            if not lyric_text:
                continue
            try:
                return LyricLineStamp(lyric_text)
            except Exception:
                continue
        return None


def search_kugou_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
) -> list[SearchCandidate]:
    return KugouProvider().search(name, duration, artist, album)
