from Lazulite.Search.Kugou import KugouProvider, search_kugou_music
from Lazulite.Search.LRCLIB import LRCLIBProvider, get_lrclib_lyric, search_lrclib_music
from Lazulite.Search.Netease import NeteaseProvider, get_163_lyric, search_163_music
from Lazulite.Search.QQMusic import QQMusicProvider, get_qq_lyric, search_qq_music
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate


def build_default_provider_registry() -> list[OnlineLyricProvider]:
    return [
        NeteaseProvider(),
        KugouProvider(),
        QQMusicProvider(),
        LRCLIBProvider(),
    ]


__all__ = [
    "KugouProvider",
    "LRCLIBProvider",
    "NeteaseProvider",
    "QQMusicProvider",
    "OnlineLyricProvider",
    "SearchCandidate",
    "build_default_provider_registry",
    "get_163_lyric",
    "get_lrclib_lyric",
    "get_qq_lyric",
    "search_163_music",
    "search_kugou_music",
    "search_lrclib_music",
    "search_qq_music",
]
