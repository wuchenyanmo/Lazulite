from Lazulite.Search.Kugou import KugouProvider, match_kugou_search_result, search_kugou_music
from Lazulite.Search.Netease import NeteaseProvider, match_163_search_result, parse_163_artist_dict, search_163_music
from Lazulite.Search.QQMusic import QQMusicProvider, match_qq_search_result, parse_qq_artist_dict, search_qq_music

__all__ = [
    "KugouProvider",
    "NeteaseProvider",
    "QQMusicProvider",
    "match_163_search_result",
    "match_kugou_search_result",
    "match_qq_search_result",
    "parse_163_artist_dict",
    "parse_qq_artist_dict",
    "search_163_music",
    "search_kugou_music",
    "search_qq_music",
]
