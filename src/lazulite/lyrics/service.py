"""在线歌词搜索、获取和转写重排的轻量组合函数。"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from lazulite.transcription.evidence import quality, comparable_confidence
from .search import _traditional_to_simplified
from typing import Iterable, Sequence
import warnings
from requests import RequestException

from lazulite.lyrics.providers import (
    FetchedLyricCandidate,
    KugouProvider,
    LRCLIBProvider,
    LyricProvider,
    NeteaseProvider,
    QQMusicProvider,
    SearchCandidate,
)
from lazulite.lyrics.normalize import normalize_lyric_text
from lazulite.lyrics.similarity import _normalized_similarity, text_similarity
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
    content_diagnostics: tuple[tuple[str, float], ...] = ()


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
    artist_variants = build_search_text_variants(metadata.artist) or [
        metadata.artist or ""
    ]
    album_variants = build_search_text_variants(metadata.album) or [
        metadata.album or ""
    ]
    all_candidates: list[SearchCandidate] = []
    failures: list[tuple[str, Exception]] = []
    successful_sources = 0

    def search_one(provider):
        failure = None
        succeeded = False
        deduplicated: dict[str, SearchCandidate] = {}
        queries: list[tuple[str, str | None, str | None]] = []
        combined_sources = {"netease", "qqmusic", "kugou"}
        for title in title_variants:
            for artist, album in (
                (artist_variants[0], album_variants[0]),
                (artist_variants[-1], album_variants[-1]),
                (artist_variants[-1], ""),
            ):
                query = (title, artist or None, None if provider.source_name in combined_sources else album or None)
                if query not in queries:
                    queries.append(query)
        if provider.source_name in {"netease", "qqmusic", "kugou"}:
            for title in title_variants:
                query = (title, None, None)
                if query not in queries: queries.append(query)
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
            except (RequestException, RuntimeError) as error:
                failure = (provider.source_name, error)
                break
            succeeded = True
            for candidate in candidates:
                if candidate.source != provider.source_name:
                    raise ValueError(
                        f"来源 {provider.source_name} 返回了错误的候选来源 {candidate.source}"
                    )
                key = candidate.candidate_id
                previous = deduplicated.get(key)
                if previous is None or candidate.match_score > previous.match_score:
                    deduplicated[key] = candidate
        return (
            sorted(
                (
                    item
                    for item in deduplicated.values()
                    if item.match_score >= min_match_score
                ),
                key=lambda item: (-item.match_score, item.candidate_id),
            )[:per_provider_limit],
            failure,
            succeeded,
        )

    with ThreadPoolExecutor(max_workers=min(len(selected), 4)) as executor:
        for candidates, failure, succeeded in executor.map(search_one, selected):
            successful_sources += int(succeeded)
            all_candidates.extend(candidates)
            if failure is not None:
                failures.append(failure)
    if not successful_sources:
        details = "; ".join(f"{source}: {error}" for source, error in failures)
        raise RuntimeError(f"所有歌词来源搜索失败: {details}") from failures[0][1]
    for source, error in failures:
        warnings.warn(f"歌词搜索部分请求失败，已保留可用候选 {source}: {error}", RuntimeWarning)
    priority = {provider.source_name: provider.priority for provider in selected}
    return tuple(
        sorted(
            all_candidates,
            key=lambda item: (
                -item.match_score,
                -priority[item.source],
                item.source,
                item.candidate_id,
            ),
        )
    )


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
            warnings.warn(
                f"没有 {candidate.source} 的歌词来源实例，已跳过候选", RuntimeWarning
            )
            continue
        ranks[candidate.source] = ranks.get(candidate.source, 0) + 1
        try:
            lyrics = provider.fetch(candidate)
        except Exception as error:
            warnings.warn(
                f"歌词获取失败，已跳过 {candidate.source}:{candidate.candidate_id}: {error}",
                RuntimeWarning,
            )
            continue
        if lyrics is None or len(lyrics.document.lines) < min_lyric_lines:
            continue
        fetched.append(
            FetchedLyricCandidate(candidate, lyrics, ranks[candidate.source])
        )
    priority = {provider.source_name: provider.priority for provider in selected}
    return tuple(
        sorted(
            fetched,
            key=lambda item: (
                -priority[item.candidate.source],
                item.provider_rank,
                -item.candidate.match_score,
            ),
        )
    )


def content_similarity(left, right):
    converter = _traditional_to_simplified()
    return max(
        text_similarity(left, right),
        text_similarity(converter.convert(left), converter.convert(right)),
    )


def _prepare_usable_segments(transcription, analysis, converter):
    by_id = {s.id: s for s in analysis.segments} if analysis else {}
    prepared = []
    for segment in transcription.segments:
        if segment.text.strip() and quality(segment.text, by_id.get(segment.source_segment_id)).usable:
            raw_norm = normalize_lyric_text(segment.text)
            simp_norm = normalize_lyric_text(converter.convert(segment.text))
            prepared.append((raw_norm, simp_norm, comparable_confidence(segment)))
    return prepared


def _content_evidence(item, transcription, analysis=None, *, usable_prepared=None, converter=None):
    converter = converter or _traditional_to_simplified()
    usable = (
        _prepare_usable_segments(transcription, analysis, converter)
        if usable_prepared is None
        else usable_prepared
    )
    lyrics = [
        (normalize_lyric_text(line.text), normalize_lyric_text(converter.convert(line.text)))
        for line in item.lyrics.document.lines
        if line.text.strip()
    ]
    matched = set()
    weighted = []
    for seg_raw, seg_simp, confidence in usable:
        if not lyrics:
            break
        best_index = -1
        best_score = -1.0
        for index, (line_raw, line_simp) in enumerate(lyrics):
            score = _normalized_similarity(line_raw, seg_raw)
            if line_simp != line_raw or seg_simp != seg_raw:
                score = max(score, _normalized_similarity(line_simp, seg_simp))
            if score > best_score:
                best_score = score
                best_index = index
        if best_score < 0.28:
            continue
        weighted.append(best_score * (confidence if confidence is not None else 1))
        matched.add(best_index)
    mean = sum(weighted) / len(weighted) if weighted else 0
    coverage = len(matched) / max(min(len(usable), len(lyrics)), 1)
    score = min(1.0, 0.70 * mean + 0.30 * coverage)
    return score, (
        ("usable_chunks", len(usable)),
        ("matched_chunks", len(weighted)),
        ("unique_lines", len(matched)),
        ("weighted_similarity", mean),
        ("unique_line_ratio", coverage),
    )


def rank_lyrics(
    candidates: Iterable[FetchedLyricCandidate],
    transcription: TranscriptionResult,
    analysis: VocalAnalysis | None = None,
    *,
    providers: Sequence[LyricProvider] | None = None,
    metadata_weight: float = 0.55,
) -> tuple[RankedLyricCandidate, ...]:
    """以通用转写正文重排歌词候选；使用通用质量证据及可解释的同类分数。"""
    if not 0 <= metadata_weight <= 1:
        raise ValueError("metadata_weight 必须在 [0, 1] 内")
    selected = tuple(providers) if providers is not None else build_default_providers()
    priority = {provider.source_name: provider.priority for provider in selected}
    converter = _traditional_to_simplified()
    usable_prepared = _prepare_usable_segments(transcription, analysis, converter)
    ranked: list[RankedLyricCandidate] = []
    for item in candidates:
        metadata_score = max(0.0, min(1.0, item.candidate.match_score / 100.0))
        content_score, diagnostics = _content_evidence(
            item, transcription, analysis, usable_prepared=usable_prepared, converter=converter
        )
        ranked.append(
            RankedLyricCandidate(
                item=item,
                metadata_score=metadata_score,
                content_score=content_score,
                final_score=metadata_weight * metadata_score
                + (1 - metadata_weight) * content_score,
                provider_priority=priority.get(item.candidate.source, 0),
                content_diagnostics=diagnostics,
            )
        )
    qualified = [item for item in ranked if item.final_score >= 0.55]
    pool = qualified if qualified else ranked
    return tuple(
        sorted(
            pool,
            key=lambda item: (
                -int(
                    any(
                        line.source_start is not None
                        for line in item.item.lyrics.document.lines
                    )
                ),
                -item.provider_priority,
                -item.final_score,
                item.item.provider_rank,
                item.item.candidate.source,
                item.item.candidate.candidate_id,
            ),
        )
    )
