from Lazulite.Search.Kugou import KugouProvider, search_kugou_music
from Lazulite.Search.Netease import NeteaseProvider, get_163_lyric, search_163_music
from Lazulite.Search.Provider import OnlineLyricProvider, SearchCandidate


def build_default_provider_registry() -> list[OnlineLyricProvider]:
    return [
        NeteaseProvider(),
        KugouProvider(),
    ]


__all__ = [
    "KugouProvider",
    "NeteaseProvider",
    "OnlineLyricProvider",
    "SearchCandidate",
    "build_default_provider_registry",
    "get_163_lyric",
    "search_163_music",
    "search_kugou_music",
]
