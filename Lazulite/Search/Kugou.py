from __future__ import annotations

import base64
import os
import warnings

from fuzzywuzzy import fuzz
from mutagen.mp4 import MP4
import numpy as np
import requests
from requests.exceptions import RequestException, SSLError
from urllib3.exceptions import InsecureRequestWarning

from Lazulite.Lyric import LyricLineStamp

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
            response = requests.get(url, params=params, headers=KUGOU_HEADERS, timeout=timeout, verify=False)
            response.raise_for_status()
            return response.json()
    except RequestException as exc:
        warnings.warn(f"酷狗请求失败，已跳过当前请求: {exc}", RuntimeWarning)
        return {}


def _set_match_score(result: dict, score: float) -> dict:
    result["match sore"] = score
    result["match_score"] = score
    result["source"] = "kugou"
    return result


def _split_kugou_artists(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = str(value)
    for sep in ("、", " / ", "/", "&", "·", ";", "；", ",", "，"):
        normalized = normalized.replace(sep, "|")
    return [item.strip() for item in normalized.split("|") if item.strip()]


def _build_kugou_keyword(result: dict) -> str:
    singer = str(result.get("singername") or "").strip()
    song = str(result.get("songname") or result.get("songname_original") or "").strip()
    if singer and song:
        return f"{singer} - {song}"
    if song:
        return song
    if singer:
        return singer
    return str(result.get("filename") or "").strip()


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
    name_weight: float = 0.7,
    album_weight: float = 0.2,
    artist_weight: float = 0.1,
    full_match_weight: float = 0.2,
    duration_threshold: float = 15.0,
) -> float:
    """
    根据歌曲名、歌手和专辑对酷狗搜索结果打分。
    """

    def fuzz_match(str1: str, str2: str) -> float:
        return fuzz.partial_ratio(str1, str2) * (1 - full_match_weight) + fuzz.ratio(str1, str2) * full_match_weight

    result_duration = float(result.get("duration") or 0.0)
    if np.abs(result_duration - duration) > duration_threshold:
        return 0.0

    candidate_names = _kugou_candidate_names(result)
    score = {"name": np.max([fuzz_match(name, item) for item in candidate_names]) if candidate_names else 0.0}

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        result_artists = _split_kugou_artists(result.get("singername"))
        score["artist"] = np.max([fuzz_match(artist, item) for item in result_artists]) if result_artists else 0.0

    if album is None:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        album_name = str(result.get("album_name") or "").strip()
        score["album"] = fuzz_match(album, album_name) if album_name else 0.0

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    return (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight


def search_kugou_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
    pagesize: int = 20,
) -> list[dict]:
    params = {
        "format": "json",
        "keyword": name,
        "page": 1,
        "pagesize": max(1, int(pagesize)),
        "showtype": 1,
    }
    res = _safe_json_get(KUGOU_SEARCH_API, params=params)
    res_list = (res.get("data") or {}).get("info", [])
    for item in res_list:
        score = match_kugou_search_result(name, duration, item, artist, album)
        _set_match_score(item, score)
    res_list.sort(key=lambda item: float(item.get("match sore", 0.0)), reverse=True)
    return res_list


def search_kugou_music_file(file: os.PathLike) -> list[dict]:
    audio = MP4(file)
    name = audio.tags["©nam"][0]
    artist = audio.tags["©ART"][0]
    album = audio.tags["©alb"][0]
    duration = audio.info.length
    return search_kugou_music(name, duration, artist, album)


def search_kugou_lyric_candidates(
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
    res = _safe_json_get(KUGOU_LYRIC_SEARCH_API, params=params)
    return list(res.get("candidates") or [])


def download_kugou_lyric(lyric_id: str | int, accesskey: str, fmt: str = "lrc") -> str | None:
    params = {
        "ver": 1,
        "client": "pc",
        "id": lyric_id,
        "accesskey": accesskey,
        "fmt": fmt,
        "charset": "utf8",
    }
    res = _safe_json_get(KUGOU_LYRIC_DOWNLOAD_API, params=params)
    encoded = res.get("content")
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def get_kugou_lyric(song_hash: str, duration: float, keyword: str | None = None) -> LyricLineStamp | None:
    """
    通过酷狗歌曲 hash 搜索歌词候选，并下载首个可用的 LRC。
    """
    duration_ms = int(round(duration * 1000))
    lyric_keyword = (keyword or "").strip() or song_hash
    candidates = search_kugou_lyric_candidates(
        keyword=lyric_keyword,
        duration_ms=duration_ms,
        hash_value=song_hash,
    )
    for candidate in candidates:
        lyric_text = download_kugou_lyric(
            lyric_id=candidate.get("id"),
            accesskey=str(candidate.get("accesskey") or ""),
            fmt="lrc",
        )
        if not lyric_text:
            continue
        try:
            return LyricLineStamp(lyric_text)
        except Exception:
            continue
    return None


def get_kugou_lyric_from_candidate(candidate: dict) -> LyricLineStamp | None:
    song_hash = str(candidate.get("hash") or "").strip()
    if not song_hash:
        return None
    duration = float(candidate.get("duration") or 0.0)
    keyword = _build_kugou_keyword(candidate)
    return get_kugou_lyric(song_hash=song_hash, duration=duration, keyword=keyword)
