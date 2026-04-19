from __future__ import annotations

from itertools import chain
import os
from urllib.parse import quote

from fuzzywuzzy import fuzz
from mutagen.mp4 import MP4
import numpy as np
import requests
from requests.exceptions import RequestException
import warnings

from Lazulite.Lyric import LyricLineStamp

HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
}
SEARCH_LIMIT = 20
SEARCH_163_API_THIRD = "https://apis.netstart.cn/music/cloudsearch?keywords={name}"
LYRIC_163_API = "https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"


def _set_match_score(result: dict, score: float) -> dict:
    result["match sore"] = score
    result["match_score"] = score
    result["source"] = "netease"
    return result


def parse_163_artist_dict(artist_dict: dict) -> list[str]:
    """
    解析网易云歌手对象，返回去重后的候选名字列表。
    """
    artist_list = [artist_dict["name"]]
    for key in ("tns", "alia", "alias"):
        values = artist_dict.get(key) or []
        artist_list.extend(values)
    return list(set(artist_list))


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

    def fuzz_match(str1: str, str2: str) -> float:
        return fuzz.partial_ratio(str1, str2) * (1 - full_match_weight) + fuzz.ratio(str1, str2) * full_match_weight

    if np.abs(result["dt"] / 1000 - duration) > duration_threshold:
        return 0.0

    result_names = [result["name"], *(result.get("alia") or [])]
    score = {"name": np.max([fuzz_match(name, res_name) for res_name in result_names])}

    if artist is None:
        artist_weight = 0.0
        score["artist"] = 0.0
    else:
        result_artists = [parse_163_artist_dict(d) for d in result.get("ar", [])]
        result_artists = list(chain(*result_artists))
        score["artist"] = np.max([fuzz_match(artist, res_artist) for res_artist in result_artists]) if result_artists else 0.0

    if album is None or "al" not in result:
        album_weight = 0.0
        score["album"] = 0.0
    else:
        result_albums = [result["al"]["name"], *(result["al"].get("tns") or [])]
        score["album"] = np.max([fuzz_match(album, res_album) for res_album in result_albums]) if result_albums else 0.0

    total_weight = name_weight + artist_weight + album_weight
    if total_weight <= 0:
        return 0.0
    return (name_weight * score["name"] + artist_weight * score["artist"] + album_weight * score["album"]) / total_weight


def search_163_music(
    name: str,
    duration: float,
    artist: str | None = None,
    album: str | None = None,
) -> list[dict]:
    url = SEARCH_163_API_THIRD.format(name=quote(name))
    try:
        response = requests.get(url, headers=HEADER, timeout=(5, 7))
        response.raise_for_status()
        res = response.json()
    except RequestException as exc:
        warnings.warn(f"网易云搜索请求失败，已跳过该来源: {exc}", RuntimeWarning)
        return []

    res_list = res.get("result", {}).get("songs", [])
    for item in res_list:
        score = match_163_search_result(name, duration, item, artist, album)
        _set_match_score(item, score)
    res_list.sort(key=lambda item: float(item.get("match sore", 0.0)), reverse=True)
    return res_list


def search_163_music_file(file: os.PathLike) -> list[dict]:
    audio = MP4(file)
    name = audio.tags["©nam"][0]
    artist = audio.tags["©ART"][0]
    album = audio.tags["©alb"][0]
    duration = audio.info.length
    return search_163_music(name, duration, artist, album)


def get_163_lyric(song_id: str | int) -> LyricLineStamp | None:
    url = LYRIC_163_API.format(song_id=song_id)
    try:
        response = requests.get(url, headers=HEADER, timeout=(5, 7))
        response.raise_for_status()
        res = response.json()
    except RequestException as exc:
        warnings.warn(f"网易云歌词请求失败，已跳过该候选: {exc}", RuntimeWarning)
        return None
    if "lrc" not in res:
        return None
    if res.get("nolyric") or res.get("pureMusic"):
        return None
    lyric_text = (res.get("lrc") or {}).get("lyric")
    if not lyric_text:
        return None

    lyric = LyricLineStamp(lyric_text)
    translation_text = (res.get("tlyric") or {}).get("lyric")
    if translation_text:
        lyric.load_translation(translation_text)
    return lyric
