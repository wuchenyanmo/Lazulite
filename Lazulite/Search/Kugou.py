from __future__ import annotations

import base64
import warnings

import numpy as np
import requests
from requests.exceptions import RequestException, SSLError
from urllib3.exceptions import InsecureRequestWarning

from Lazulite.Lyric import LyricLineStamp
from Lazulite.Search.Common import combined_fuzzy_score
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate

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
    if not value:
        return []
    normalized = str(value)
    for sep in ("、", " / ", "/", "&", "·", ";", "；", ",", "，"):
        normalized = normalized.replace(sep, "|")
    return [item.strip() for item in normalized.split("|") if item.strip()]


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


def match_kugou_search_result(
    name: str,
    duration: float,
    result: dict,
    artist: str | None = None,
    album: str | None = None,
    full_match_weight: float = 0.2,
    name_weight: float = 0.7,
    album_weight: float = 0.2,
    artist_weight: float = 0.1,
    duration_threshold: float = 15.0,
) -> float:
    result_duration = float(result.get("duration") or 0.0)
    if np.abs(result_duration - duration) > duration_threshold:
        return 0.0

    candidate_names = _kugou_candidate_names(result)
    score = {
        "name": max(combined_fuzzy_score(name, item, full_match_weight=full_match_weight) for item in candidate_names)
        if candidate_names else 0.0,
    }

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        result_artists = _split_kugou_artists(result.get("singername"))
        score["artist"] = (
            max(combined_fuzzy_score(artist, item, full_match_weight=full_match_weight) for item in result_artists)
            if result_artists else 0.0
        )

    if album is None:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        album_name = str(result.get("album_name") or "").strip()
        score["album"] = combined_fuzzy_score(album, album_name, full_match_weight=full_match_weight) if album_name else 0.0

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    value = (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight
    return float(value)


class KugouProvider(OnlineLyricProvider):
    source_name = "kugou"

    def search(
        self,
        title: str,
        duration: float,
        artist: str | None = None,
        album: str | None = None,
        pagesize: int = 20,
    ) -> list[SearchCandidate]:
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
            candidates.append(
                SearchCandidate(
                    source=self.source_name,
                    candidate_id=str(item.get("hash") or ""),
                    title=str(item.get("songname") or item.get("songname_original") or "").strip(),
                    artist=artist_name,
                    album=str(item.get("album_name") or "").strip() or None,
                    duration=float(item.get("duration") or 0.0),
                    match_score=match_kugou_search_result(title, duration, item, artist, album),
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
