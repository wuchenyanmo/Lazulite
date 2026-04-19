from Lazulite.Search.Kugou import (
    download_kugou_lyric,
    get_kugou_lyric,
    get_kugou_lyric_from_candidate,
    search_kugou_lyric_candidates,
    search_kugou_music,
    search_kugou_music_file,
)
from Lazulite.Search.Netease import (
    get_163_lyric,
    match_163_search_result,
    parse_163_artist_dict,
    search_163_music,
    search_163_music_file,
)

__all__ = [
    "download_kugou_lyric",
    "get_163_lyric",
    "get_kugou_lyric",
    "get_kugou_lyric_from_candidate",
    "match_163_search_result",
    "parse_163_artist_dict",
    "search_163_music",
    "search_163_music_file",
    "search_kugou_lyric_candidates",
    "search_kugou_music",
    "search_kugou_music_file",
]
