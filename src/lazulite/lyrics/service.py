"""在线歌词搜索、获取和转写重排的轻量组合函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

from lazulite.lyrics.providers import (
    FetchedLyricCandidate,
    KugouProvider,
    LRCLIBProvider,
    LyricProvider,
    NeteaseProvider,
    QQMusicProvider,
    SearchCandidate,
)
from lazulite.lyrics.similarity import text_similarity
from lazulite.models.audio import AudioMetadata
from lazulite.models.audio import VocalAnalysis
from lazulite.models.transcription import TranscriptionResult

from .search import build_search_text_variants


@dataclass(frozen=True, slots=True)
class RankedLyricCandidate:
    """候选重排的可审计分数，不把模型特定字段泄漏到搜索层。"""

    item: FetchedLyricCandidate
    metadata_score: float
    content_score: float
    final_score: float
    provider_priority: int


def build_default_providers() -> tuple[LyricProvider, ...]:
    """按稳定优先级创建来源实例；调用方也可以完全自行注入。"""
    return (NeteaseProvider(), QQMusicProvider(), KugouProvider(), LRCLIBProvider())


def _provider_map(providers: Iterable[LyricProvider]) -> dict[str, LyricProvider]:
    result: dict[str, LyricProvider] = {}
    for provider in providers:
        if provider.source_name in result:
            raise ValueError(f"歌词来源名称重复: {provider.source_name}")
        result[provider.source_name] = provider
    return result


def search_lyrics(
    metadata: AudioMetadata,
    *,
    providers: Sequence[LyricProvider] | None = None,
    min_match_score: float = 0.0,
    per_provider_limit: int = 3,
) -> tuple[SearchCandidate, ...]:
    """使用元数据搜索所有来源，并以来源内去重后的候选返回。

    网络错误只跳过发生错误的来源；若每个来源都抛出错误则保留失败信号，
    从而区别于正常的零搜索结果。
    """
    if not metadata.title:
        raise ValueError("音频元数据缺少标题，无法搜索歌词")
    if per_provider_limit < 1:
        raise ValueError("per_provider_limit 必须为正数")
    selected = tuple(providers) if providers is not None else build_default_providers()
    if not selected:
        return ()
    _provider_map(selected)
    title_variants = build_search_text_variants(metadata.title)
    artist_variants = build_search_text_variants(metadata.artist) or [metadata.artist or ""]
    album_variants = build_search_text_variants(metadata.album) or [metadata.album or ""]
    all_candidates: list[SearchCandidate] = []
    failures: list[tuple[str, Exception]] = []
    for provider in selected:
        deduplicated: dict[str, SearchCandidate] = {}
        queries: list[tuple[str, str | None, str | None]] = []
        for title in title_variants:
            for artist, album in ((artist_variants[0], album_variants[0]), (artist_variants[-1], album_variants[-1]), (artist_variants[-1], "")):
                query = (title, artist or None, album or None)
                if query not in queries:
                    queries.append(query)
        for title, artist, album in queries:
            try:
                candidates = provider.search(
                    title=title,
                    duration=metadata.duration,
                    artist=artist,
                    album=album,
                    score_title=metadata.title,
                    score_artist=metadata.artist,
                    score_album=metadata.album,
                )
            except Exception as error:  # 单个网络来源不可阻断其他来源。
                failures.append((provider.source_name, error))
                break
            for candidate in candidates:
                if candidate.source != provider.source_name:
                    raise ValueError(f"来源 {provider.source_name} 返回了错误的候选来源 {candidate.source}")
                key = candidate.candidate_id
                previous = deduplicated.get(key)
                if previous is None or candidate.match_score > previous.match_score:
                    deduplicated[key] = candidate
        all_candidates.extend(
            sorted(
                (item for item in deduplicated.values() if item.match_score >= min_match_score),
                key=lambda item: (-item.match_score, item.candidate_id),
            )[:per_provider_limit]
        )
    if len(failures) == len(selected):
        details = "; ".join(f"{source}: {error}" for source, error in failures)
        raise RuntimeError(f"所有歌词来源搜索失败: {details}") from failures[0][1]
    for source, error in failures:
        warnings.warn(f"歌词搜索来源失败，已跳过 {source}: {error}", RuntimeWarning)
    priority = {provider.source_name: provider.priority for provider in selected}
    return tuple(sorted(all_candidates, key=lambda item: (-item.match_score, -priority[item.source], item.source, item.candidate_id)))


def fetch_lyrics(
    candidates: Iterable[SearchCandidate],
    *,
    providers: Sequence[LyricProvider] | None = None,
    min_lyric_lines: int = 1,
) -> tuple[FetchedLyricCandidate, ...]:
    """获取候选歌词；空歌词和来源失败均不伪装成有效结果。"""
    if min_lyric_lines < 1:
        raise ValueError("min_lyric_lines 必须为正数")
    selected = tuple(providers) if providers is not None else build_default_providers()
    by_source = _provider_map(selected)
    ranks: dict[str, int] = {}
    fetched: list[FetchedLyricCandidate] = []
    for candidate in candidates:
        provider = by_source.get(candidate.source)
        if provider is None:
            warnings.warn(f"没有 {candidate.source} 的歌词来源实例，已跳过候选", RuntimeWarning)
            continue
        ranks[candidate.source] = ranks.get(candidate.source, 0) + 1
        try:
            lyrics = provider.fetch(candidate)
        except Exception as error:
            warnings.warn(f"歌词获取失败，已跳过 {candidate.source}:{candidate.candidate_id}: {error}", RuntimeWarning)
            continue
        if lyrics is None or len(lyrics.document.lines) < min_lyric_lines:
            continue
        fetched.append(FetchedLyricCandidate(candidate, lyrics, ranks[candidate.source]))
    priority = {provider.source_name: provider.priority for provider in selected}
    return tuple(sorted(fetched, key=lambda item: (-priority[item.candidate.source], item.provider_rank, -item.candidate.match_score)))


def _content_score(item: FetchedLyricCandidate, transcription: TranscriptionResult) -> float:
    lyric_lines = tuple(line.text for line in item.lyrics.document.lines if line.text.strip())
    transcribed = tuple(segment.text for segment in transcription.segments if segment.text.strip())
    if not lyric_lines or not transcribed:
        return 0.0
    # 覆盖率避免长文本拼接对重复副歌产生过强偏置，整体相似度保留顺序信号。
    coverage = sum(max(text_similarity(line, segment) for segment in transcribed) for line in lyric_lines) / len(lyric_lines)
    ordered = text_similarity("\n".join(lyric_lines), "\n".join(transcribed))
    return 0.65 * coverage + 0.35 * ordered


def rank_lyrics(
    candidates: Iterable[FetchedLyricCandidate],
    transcription: TranscriptionResult,
    analysis: VocalAnalysis | None = None,
    *,
    providers: Sequence[LyricProvider] | None = None,
    metadata_weight: float = 0.55,
) -> tuple[RankedLyricCandidate, ...]:
    """以通用转写正文重排歌词候选；预留声学分析参数且不混入模型置信度。"""
    if not 0 <= metadata_weight <= 1:
        raise ValueError("metadata_weight 必须在 [0, 1] 内")
    selected = tuple(providers) if providers is not None else build_default_providers()
    priority = {provider.source_name: provider.priority for provider in selected}
    ranked: list[RankedLyricCandidate] = []
    for item in candidates:
        metadata_score = max(0.0, min(1.0, item.candidate.match_score / 100.0))
        content_score = _content_score(item, transcription)
        ranked.append(RankedLyricCandidate(
            item=item,
            metadata_score=metadata_score,
            content_score=content_score,
            final_score=metadata_weight * metadata_score + (1 - metadata_weight) * content_score,
            provider_priority=priority.get(item.candidate.source, 0),
        ))
    return tuple(sorted(ranked, key=lambda item: (-item.final_score, -item.provider_priority, item.item.provider_rank)))
