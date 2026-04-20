from __future__ import annotations

import argparse
import copy
import io
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError, TXXX, USLT
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

from Lazulite import LyricAligner, OffsetAligner, VocalAnalyzer, WhisperTranscriber
from Lazulite.Debug import (
    load_alignment_lyric,
    save_alignment_json,
    save_offset_debug_json,
    save_transcription_json,
)
from Lazulite.Lyric import LyricLineStamp, LyricTokenLine
from Lazulite.Search.Common import build_search_text_variants
from Lazulite.Search import NeteaseProvider, build_default_provider_registry


@dataclass
class AudioMetadata:
    path: Path
    title: str | None
    artist: str | None
    album: str | None
    duration: float
    raw_tags: dict[str, Any]


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return None
    if hasattr(value, "text"):
        return _first_text(value.text)
    text = str(value).strip()
    return text or None


def read_audio_metadata(audio_path: str | Path) -> AudioMetadata:
    path = Path(audio_path).expanduser().resolve()
    audio = MutagenFile(str(path), easy=False)
    if audio is None or getattr(audio, "info", None) is None:
        raise ValueError(f"无法读取音频文件或不支持的格式: {path}")

    tags = getattr(audio, "tags", None) or {}
    suffix = path.suffix.lower()

    title = None
    artist = None
    album = None

    if suffix in {".m4a", ".mp4", ".aac", ".alac"}:
        title = _first_text(tags.get("\xa9nam"))
        artist = _first_text(tags.get("\xa9ART")) or _first_text(tags.get("aART"))
        album = _first_text(tags.get("\xa9alb"))
    elif suffix == ".mp3":
        title = _first_text(tags.get("TIT2"))
        artist = _first_text(tags.get("TPE1")) or _first_text(tags.get("TPE2"))
        album = _first_text(tags.get("TALB"))
    else:
        title = _first_text(tags.get("title"))
        artist = _first_text(tags.get("artist")) or _first_text(tags.get("albumartist"))
        album = _first_text(tags.get("album"))

    if not title:
        title = path.stem

    return AudioMetadata(
        path=path,
        title=title,
        artist=artist,
        album=album,
        duration=float(audio.info.length),
        raw_tags=dict(tags),
    )


def search_lyric_from_metadata(
    metadata: AudioMetadata,
    netease_song_id: str | int | None = None,
    min_search_score: float = 55.0,
    search_workers: int = 4,
) -> tuple[LyricLineStamp, dict[str, Any]]:
    if netease_song_id is not None:
        provider = NeteaseProvider()
        lyric = provider.fetch_lyric_by_song_id(netease_song_id)
        if lyric is None:
            raise RuntimeError(f"未能获取 netease_song_id={netease_song_id} 对应歌词")
        return lyric, {
            "source": provider.source_name,
            "candidate_id": str(netease_song_id),
            "title": metadata.title,
            "artist": metadata.artist,
            "album": metadata.album,
            "match_score": None,
        }

    if not metadata.title:
        raise ValueError("音频元数据中缺少标题，无法自动搜索歌词；请传入 --title 或 --netease-song-id")

    best_score_by_source: dict[str, float] = {}
    top_candidate_by_source: dict[str, dict[str, Any]] = {}
    attempted_by_source: dict[str, list[str]] = {}
    providers = build_default_provider_registry()
    provider_by_source = {
        provider.source_name: provider
        for provider in providers
    }
    title_variants = build_search_text_variants(metadata.title)
    artist_variants = build_search_text_variants(metadata.artist)
    album_variants = build_search_text_variants(metadata.album)
    if not artist_variants:
        artist_variants = [metadata.artist or ""]
    if not album_variants:
        album_variants = [metadata.album or ""]

    qualified_candidates_all: list[Any] = []

    def _search_provider_candidates(provider):
        deduped_candidates: dict[str, Any] = {}
        for title_variant in title_variants:
            query_artist = artist_variants[0] or None
            query_album = album_variants[0] or None
            query_pairs = [
                (title_variant, query_artist, query_album),
            ]
            if query_album:
                query_pairs.append((title_variant, query_artist, None))
            simplified_artist = artist_variants[-1] or None
            simplified_album = album_variants[-1] or None
            fallback_pair = (title_variant, simplified_artist, simplified_album)
            if fallback_pair not in query_pairs:
                query_pairs.append(fallback_pair)
            if simplified_album:
                fallback_no_album = (title_variant, simplified_artist, None)
                if fallback_no_album not in query_pairs:
                    query_pairs.append(fallback_no_album)

            for query_title, query_artist_value, query_album_value in query_pairs:
                if not query_title:
                    continue
                candidates = provider.search(
                    title=query_title,
                    duration=metadata.duration,
                    artist=query_artist_value,
                    album=query_album_value,
                    score_title=metadata.title,
                    score_artist=metadata.artist,
                    score_album=metadata.album,
                )
                for candidate in candidates:
                    key = f"{candidate.source}:{candidate.candidate_id}"
                    existing = deduped_candidates.get(key)
                    if existing is None or float(candidate.match_score) > float(existing.match_score):
                        deduped_candidates[key] = candidate

        return provider.source_name, sorted(
            deduped_candidates.values(),
            key=lambda item: float(item.match_score),
            reverse=True,
        )

    max_search_workers = max(1, min(int(search_workers), len(providers)))
    with ThreadPoolExecutor(max_workers=max_search_workers, thread_name_prefix="provider-search") as executor:
        provider_results = list(executor.map(_search_provider_candidates, providers))

    for source_name, candidates in provider_results:
        if not candidates:
            attempted_by_source[source_name] = []
            continue

        best_score_by_source[source_name] = float(candidates[0].match_score)
        top_candidate_by_source[source_name] = candidates[0].to_dict()
        qualified_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.match_score) >= min_search_score
        ]
        attempted_by_source[source_name] = []
        qualified_candidates_all.extend(qualified_candidates)

    provider_order = {
        provider.source_name: index
        for index, provider in enumerate(providers)
    }
    qualified_candidates_all.sort(
        key=lambda candidate: (
            -float(candidate.match_score),
            provider_order.get(candidate.source, len(provider_order)),
        )
    )

    for candidate in qualified_candidates_all:
        attempted_by_source.setdefault(candidate.source, []).append(
            f"{candidate.candidate_id}({candidate.match_score:.1f})"
        )
        provider = provider_by_source[candidate.source]
        try:
            lyric = provider.fetch_lyric(candidate)
        except Exception as exc:
            warnings.warn(
                f"在线歌词候选获取失败，已跳过 source={provider.source_name} candidate={candidate.candidate_id}: {exc}",
                RuntimeWarning,
            )
            lyric = None
        if lyric is not None and not getattr(lyric, "lyric_lines", []):
            warnings.warn(
                "在线歌词候选返回空歌词，已跳过 "
                f"source={provider.source_name} candidate={candidate.candidate_id}",
                RuntimeWarning,
            )
            lyric = None
        if lyric is not None:
            return lyric, candidate.to_dict()

    if best_score_by_source:
        summary = ", ".join(
            f"{source}={score:.1f}"
            for source, score in best_score_by_source.items()
        )
        candidate_summary = "; ".join(
            (
                f"{source}: "
                f"id={candidate['candidate_id']}, "
                f"title={candidate['title']}, "
                f"artist={candidate['artist']}, "
                f"album={candidate['album']}, "
                f"duration={candidate['duration']}, "
                f"match_score={float(candidate['match_score']):.1f}"
            )
            for source, candidate in top_candidate_by_source.items()
        )
        if all(score < min_search_score for score in best_score_by_source.values()):
            raise RuntimeError(
                "各平台搜索命中分数都低于阈值: "
                f"{summary} < {min_search_score:.1f}；"
                f"最高分候选: {candidate_summary}；"
                "请改用 --netease-song-id 或手动提供 --lyric-path"
            )
    else:
        candidate_summary = ""

    attempted_fragments = [
        f"{source}: {', '.join(values)}"
        for source, values in attempted_by_source.items()
        if values
    ]
    if attempted_fragments:
        raise RuntimeError(
            "已尝试所有分数达标的在线歌词候选，但都未能获取歌词；"
            f"尝试过的候选: {'; '.join(attempted_fragments)}"
            + (f"；各平台最高分候选: {candidate_summary}" if candidate_summary else "")
        )
    raise RuntimeError("歌词搜索没有返回任何可用候选结果")


def _format_lrc_timestamp(seconds: float) -> str:
    minute = int(seconds // 60)
    second = seconds % 60
    return f"[{minute:02d}:{second:05.2f}]"


def _format_lyric_line_text(line: LyricTokenLine, include_translation: bool, translation_brackets: str) -> str:
    text = line.text
    if line.singer:
        text = f"{line.singer}: {text}"
    if include_translation and line.translation:
        text = f"{text}{translation_brackets[0]}{line.translation}{translation_brackets[1]}"
    return text


def _estimate_average_alignment_offset(alignment_result) -> float:
    offsets = [
        float(item.start - item.line.timestamp)
        for item in alignment_result.items
        if item.start is not None and item.line.timestamp is not None
    ]
    if not offsets:
        return 0.0
    return sum(offsets) / len(offsets)


def _estimate_plain_text_step(alignment_result) -> float | None:
    points = [
        (item.line_index, float(item.start))
        for item in alignment_result.items
        if item.start is not None
    ]
    if len(points) < 2:
        return None
    steps = []
    for idx in range(1, len(points)):
        prev_index, prev_time = points[idx - 1]
        current_index, current_time = points[idx]
        line_gap = current_index - prev_index
        if line_gap <= 0:
            continue
        time_gap = current_time - prev_time
        if time_gap <= 0:
            continue
        steps.append(time_gap / line_gap)
    if not steps:
        return None
    steps.sort()
    return float(steps[len(steps) // 2])


def _fill_plain_text_unmatched_timestamps(alignment_result) -> list[float | None]:
    filled = [
        None if item.start is None else float(item.start)
        for item in alignment_result.items
    ]
    known_indices = [idx for idx, value in enumerate(filled) if value is not None]
    if not known_indices:
        return filled

    default_step = _estimate_plain_text_step(alignment_result) or 2.0

    first_idx = known_indices[0]
    for idx in range(first_idx - 1, -1, -1):
        next_time = filled[idx + 1]
        if next_time is None:
            continue
        filled[idx] = max(0.0, float(next_time - default_step))

    last_idx = known_indices[-1]
    for idx in range(last_idx + 1, len(filled)):
        prev_time = filled[idx - 1]
        if prev_time is None:
            continue
        filled[idx] = float(prev_time + default_step)

    for start_idx, end_idx in zip(known_indices, known_indices[1:]):
        left_time = filled[start_idx]
        right_time = filled[end_idx]
        if left_time is None or right_time is None:
            continue
        gap = end_idx - start_idx
        if gap <= 1:
            continue
        step = (right_time - left_time) / gap
        if step <= 0:
            step = default_step
        for idx in range(start_idx + 1, end_idx):
            filled[idx] = float(left_time + step * (idx - start_idx))
    return filled


def build_aligned_lrc(
    lyric: LyricLineStamp,
    alignment_result,
    include_translation: bool = True,
    translation_brackets: str = "【】",
    prefer_original_on_unmatched: bool = True,
    skip_unmatched: bool = False,
) -> str:
    lines: list[str] = []
    has_real_timestamps = getattr(lyric, "has_real_timestamps", True)
    average_offset = _estimate_average_alignment_offset(alignment_result) if has_real_timestamps else None
    plain_text_fallback_times = _fill_plain_text_unmatched_timestamps(alignment_result) if not has_real_timestamps else None

    for key in lyric.metadata_keys:
        value = lyric.metadata[key]
        lines.append(f"[{key}:{value}]")

    for idx, item in enumerate(alignment_result.items):
        timestamp = item.start
        if timestamp is None and has_real_timestamps and prefer_original_on_unmatched and item.line.timestamp is not None:
            timestamp = max(0.0, float(item.line.timestamp + float(average_offset or 0.0)))
        if timestamp is None and not has_real_timestamps and plain_text_fallback_times is not None:
            timestamp = plain_text_fallback_times[idx]
        if timestamp is None:
            if skip_unmatched:
                continue
            continue
        text = _format_lyric_line_text(item.line, include_translation, translation_brackets)
        lines.append(f"{_format_lrc_timestamp(timestamp)}{text}")

    return "\n".join(lines).strip()


def _segment_offset_priority(segment: dict) -> float:
    scores = segment.get("scores", {})
    presence = float(scores.get("vocal_presence", 0.0))
    off_center = float(scores.get("off_center_risk", 0.0))
    duration = float(segment.get("core_duration", segment.get("duration", 0.0)))
    return 1.10 * presence - 0.55 * off_center + 0.05 * min(duration, 8.0)


def select_offset_segment_indices(analysis_result) -> set[int]:
    if hasattr(analysis_result, "to_dict"):
        data = analysis_result.to_dict()
    else:
        data = analysis_result

    segments = data.get("segments", [])
    sections = data.get("sections", [])
    selected: set[int] = set()

    for section in sections:
        candidates = []
        for segment_index in section.get("segment_indices", []):
            if segment_index < 1 or segment_index > len(segments):
                continue
            segment = segments[segment_index - 1]
            scores = segment.get("scores", {})
            if float(scores.get("vocal_presence", 0.0)) < 0.48:
                continue
            if float(scores.get("off_center_risk", 0.0)) > 0.50:
                continue
            if float(segment.get("duration", 0.0)) < 1.0:
                continue
            candidates.append((_segment_offset_priority(segment), int(segment_index)))

        candidates.sort(reverse=True)
        keep_count = min(3, len(candidates))
        if keep_count <= 0:
            continue
        for _, segment_index in candidates[:keep_count]:
            selected.add(segment_index)

    if selected:
        return selected

    for index, segment in enumerate(segments, start=1):
        scores = segment.get("scores", {})
        if float(scores.get("vocal_presence", 0.0)) < 0.52:
            continue
        if float(scores.get("off_center_risk", 0.0)) > 0.45:
            continue
        selected.add(index)
    return selected


def _offset_result_is_reliable(alignment_result) -> bool:
    return bool(getattr(alignment_result, "details", {}).get("is_reliable"))


def _print_offset_alignment_summary(alignment_result) -> None:
    details = getattr(alignment_result, "details", {}) or {}
    merged_sections = details.get("merged_sections", [])
    section_offsets = details.get("sections", [])

    print(f"  merged_sections={len(merged_sections)}")
    if not section_offsets:
        print("  offsets=[]")
        return

    if len(section_offsets) == 1:
        offset = float(section_offsets[0].get("offset", 0.0))
        print(f"  global_offset={offset:+.3f}s")
        return

    print("  section_offsets:")
    for item in section_offsets:
        offset = float(item.get("offset", 0.0))
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        source_sections = item.get("source_section_indices", [])
        print(
            "   "
            f"lines={line_start}-{line_end} "
            f"offset={offset:+.3f}s "
            f"anchors={item.get('anchor_count')} "
            f"source_sections={source_sections}"
        )


def write_lyric_metadata(audio_path: str | Path, lyric_text: str) -> None:
    path = Path(audio_path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix in {".m4a", ".mp4", ".aac", ".alac"}:
        audio = MP4(str(path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags["\xa9lyr"] = [lyric_text]
        audio.save()
        return

    if suffix == ".mp3":
        try:
            audio = ID3(str(path))
        except ID3NoHeaderError:
            audio = ID3()
        audio.delall("USLT")
        audio.delall("TXXX:LYRICS")
        audio.add(USLT(encoding=3, lang="XXX", desc="", text=lyric_text))
        audio.add(TXXX(encoding=3, desc="LYRICS", text=[lyric_text]))
        audio.save(str(path), v2_version=3)
        return

    if suffix == ".flac":
        audio = FLAC(str(path))
        audio["LYRICS"] = lyric_text
        audio.save()
        return

    if suffix in {".ogg", ".oga"}:
        try:
            audio = OggVorbis(str(path))
        except Exception:
            audio = OggOpus(str(path))
        audio["LYRICS"] = [lyric_text]
        audio.save()
        return

    audio = MutagenFile(str(path), easy=False)
    if audio is None:
        raise ValueError(f"不支持写入歌词标签的文件格式: {path.suffix}")
    if getattr(audio, "tags", None) is None and hasattr(audio, "add_tags"):
        audio.add_tags()
    if getattr(audio, "tags", None) is None:
        raise ValueError(f"当前文件格式没有可写标签容器: {path.suffix}")
    audio.tags["LYRICS"] = [lyric_text]
    audio.save()


SUPPORTED_AUDIO_SUFFIXES = {
    ".m4a", ".mp4", ".aac", ".alac", ".mp3", ".flac", ".ogg", ".oga",
    ".wav", ".wave", ".opus", ".m4b",
}


def _iter_audio_files_in_directory(directory: Path) -> list[Path]:
    files = [
        path for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return files


def _batch_log_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "lazulite_batch.log", directory / "lazulite_batch.err"


class _TeeStream:
    def __init__(self, mirror, buffer: io.StringIO):
        self._mirror = mirror
        self._buffer = buffer

    def write(self, data):
        self._mirror.write(data)
        self._buffer.write(data)
        return len(data)

    def flush(self):
        self._mirror.flush()
        self._buffer.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取音频元数据，搜索歌词，执行分片转写与约束对齐，并将对齐歌词写回音频标签。"
    )
    parser.add_argument("audio_path", help="输入音频文件路径")
    parser.add_argument("--netease-song-id", help="直接指定网易云歌曲 ID，跳过在线搜索")
    parser.add_argument("--title", help="覆盖音频标题元数据")
    parser.add_argument("--artist", help="覆盖音频歌手元数据")
    parser.add_argument("--album", help="覆盖音频专辑元数据")
    parser.add_argument("--lyric-path", help="本地歌词文件路径；提供后优先使用本地歌词而不是网络搜索")
    parser.add_argument(
        "--read-metadata-lyric",
        action="store_true",
        help="未提供 --lyric-path 时，优先从音频元数据中的内嵌歌词读取 LRC/纯文本歌词",
    )
    parser.add_argument("--language", help="Whisper 语言参数，例如 zh / ja / en")
    parser.add_argument("--model-id", default="openai/whisper-large-v3", help="Whisper 模型目录或模型名")
    parser.add_argument(
        "--align-mode",
        default="auto",
        choices=["auto", "offset-only", "dp"],
        help="对齐模式：auto 先尝试轻量 offset，不足时回退到 token+DP；offset-only 仅做偏移估计；dp 始终走 token 时间戳 + 动态规划",
    )
    parser.add_argument("--prompt-mode", default="hybrid", choices=["none", "previous", "hint", "hybrid"])
    parser.add_argument("--num-candidates", type=int, default=1, help="每个分片重复转写次数")
    parser.add_argument("--min-search-score", type=float, default=55.0, help="歌词搜索最低接受分数")
    parser.add_argument("--search-workers", type=int, default=4, help="在线歌词 provider 搜索并发数")
    parser.add_argument("--output-lrc", help="将最终对齐歌词额外保存为 .lrc 文件")
    parser.add_argument("--transcription-json", help="将分片转写结果额外保存为 JSON")
    parser.add_argument("--alignment-json", help="将对齐结果额外保存为 JSON")
    parser.add_argument("--offset-debug-json", help="将 offset 路径的调试信息额外保存为 JSON")
    parser.add_argument(
        "--low-memory-mode",
        action="store_true",
        help="启用低显存模式，在阶段间主动卸载 Demucs/Whisper 并清理显存",
    )
    parser.add_argument(
        "--skip-unmatched",
        action="store_true",
        help="输出 LRC 时跳过未成功对齐的歌词行；默认保留并使用已对齐行的平均时间偏移推断时间戳",
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="写回歌词时不包含翻译文本",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _resolve_lyric_source(
    args: argparse.Namespace,
    audio_path: Path,
    metadata: AudioMetadata,
) -> tuple[LyricLineStamp, dict[str, Any]]:
    if args.lyric_path or args.read_metadata_lyric:
        lyric = load_alignment_lyric(args.lyric_path, music_path=str(audio_path))
        if lyric is None and args.lyric_path:
            raise RuntimeError(f"无法从本地歌词文件加载歌词: {args.lyric_path}")
        if lyric is not None:
            return lyric, {
                "source": "metadata" if args.read_metadata_lyric and not args.lyric_path else "local",
                "candidate_id": None,
                "title": metadata.title,
                "artist": metadata.artist,
                "album": metadata.album,
                "match_score": None,
            }

    return search_lyric_from_metadata(
        metadata=metadata,
        netease_song_id=args.netease_song_id,
        min_search_score=args.min_search_score,
        search_workers=args.search_workers,
    )


def process_audio_file(args: argparse.Namespace, audio_path: str | Path) -> None:
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    metadata = read_audio_metadata(audio_path)
    if args.title:
        metadata.title = args.title
    if args.artist:
        metadata.artist = args.artist
    if args.album:
        metadata.album = args.album

    print("音频元数据:")
    print(f"  path={metadata.path}")
    print(f"  title={metadata.title}")
    print(f"  artist={metadata.artist}")
    print(f"  album={metadata.album}")
    print(f"  duration={metadata.duration:.2f}s")

    analyzer = VocalAnalyzer(low_memory_mode=args.low_memory_mode)
    aligner = LyricAligner()
    offset_aligner = OffsetAligner()
    lyric_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lyric-search")
    lyric_future: Future[tuple[LyricLineStamp, dict[str, Any]]] = lyric_executor.submit(
        _resolve_lyric_source,
        args,
        audio_path,
        metadata,
    )
    offset_transcriber = None
    dp_transcriber = None
    track_result = None
    alignment_result = None
    offset_track_result = None
    offset_alignment_result = None
    offset_segment_indices: set[int] = set()
    offset_fallback_to = None
    lyric: LyricLineStamp | None = None
    search_result: dict[str, Any] | None = None

    try:
        analysis_result = analyzer.analyze_file(str(audio_path))
        print("人声分析:")
        print(f"  is_vocal={analysis_result.is_vocal}")
        print(f"  vocal_time={analysis_result.vocal_time:.2f}s")
        print(f"  sections={len(analysis_result.sections)}")
        print(f"  segments={len(analysis_result.segments)}")

        if not analysis_result.is_vocal:
            print("警告:")
            print("  当前音频未检测到明显人声，已跳过转写、对齐、歌词写回与歌词文件输出")
            return

        analyzer.release_model_if_needed()

        lyric, search_result = lyric_future.result()
        print("歌词来源:")
        print(f"  source={search_result.get('source', 'netease')}")
        print(f"  candidate_id={search_result.get('candidate_id')}")
        print(f"  title={search_result.get('title')}")
        print(f"  artist={search_result.get('artist')}")
        print(f"  album={search_result.get('album')}")
        if search_result.get("duration") is not None:
            print(f"  candidate_duration={float(search_result['duration']):.2f}s")
        print(f"  match_score={search_result.get('match_score')}")
        print(f"  lyric_lines={len(lyric.lyric_lines)}")

        lyric_has_real_timestamps = getattr(lyric, "has_real_timestamps", True)

        if args.align_mode == "offset-only" and not lyric_has_real_timestamps:
            raise RuntimeError("offset-only 模式要求输入歌词包含真实时间戳；当前歌词为纯文本，请改用 auto 或 dp")

        if args.align_mode in {"auto", "offset-only"} and lyric_has_real_timestamps:
            offset_segment_indices = select_offset_segment_indices(analysis_result)
            print("Offset 预筛选:")
            print(f"  candidate_segments={len(offset_segment_indices)}")

            offset_transcriber = WhisperTranscriber(
                model_id=args.model_id,
                language=args.language,
                prompt_mode="previous",
                num_candidates=args.num_candidates,
                enable_token_timestamps=False,
                low_memory_mode=args.low_memory_mode,
            )
            track_result = offset_transcriber.transcribe_analysis(
                analysis_result=analysis_result,
                lyric_hint=None,
                segment_indices=offset_segment_indices,
            )
            offset_track_result = track_result
            print("Offset 分片转写:")
            print(f"  chunks={len(track_result.chunks)}")
            print(f"  stats={track_result.stats}")

            alignment_result = offset_aligner.align(lyric=lyric, transcription=track_result)
            offset_alignment_result = alignment_result
            print("Offset 对齐:")
            print(f"  stats={alignment_result.stats}")
            print(f"  reliable={_offset_result_is_reliable(alignment_result)}")
            _print_offset_alignment_summary(alignment_result)

            if args.offset_debug_json:
                save_offset_debug_json(
                    Path(args.offset_debug_json).expanduser().resolve(),
                    analysis_result.to_dict(),
                    lyric=lyric,
                    track_result=offset_track_result,
                    alignment_result=offset_alignment_result,
                    selected_segment_indices=offset_segment_indices,
                    mode=args.align_mode,
                    fallback_to=offset_fallback_to,
                )
                print(f"已保存 Offset 调试 JSON: {args.offset_debug_json}")

            if args.align_mode == "offset-only" and not _offset_result_is_reliable(alignment_result):
                print("警告:")
                print("  offset-only 模式的 anchor 数量不足或一致性不足，继续忽略可靠性阈值计算全局 offset")
                alignment_result = offset_aligner.align(
                    lyric=lyric,
                    transcription=track_result,
                    force_global_fallback=True,
                    fallback_to_original_on_no_anchor=True,
                )
                offset_alignment_result = alignment_result
                print("Offset 强制回退:")
                print(f"  stats={alignment_result.stats}")
                _print_offset_alignment_summary(alignment_result)

                if args.offset_debug_json:
                    save_offset_debug_json(
                        Path(args.offset_debug_json).expanduser().resolve(),
                        analysis_result.to_dict(),
                        lyric=lyric,
                        track_result=offset_track_result,
                        alignment_result=offset_alignment_result,
                        selected_segment_indices=offset_segment_indices,
                        mode=args.align_mode,
                        fallback_to="forced-global-or-original",
                    )
                    print(f"已更新 Offset 调试 JSON: {args.offset_debug_json}")

        if args.align_mode == "dp" or (args.align_mode == "auto" and (not lyric_has_real_timestamps or not _offset_result_is_reliable(alignment_result))):
            if args.align_mode == "auto":
                print("Offset 回退:")
                if not lyric_has_real_timestamps:
                    print("  reason=歌词不包含真实时间戳，跳过 offset，切换到 token 时间戳 + 动态规划")
                else:
                    print("  reason=anchor 不足或 section 偏移一致性不足，切换到 token 时间戳 + 动态规划")
                offset_fallback_to = "dp"

                if args.offset_debug_json and offset_track_result is not None and offset_alignment_result is not None:
                    save_offset_debug_json(
                        Path(args.offset_debug_json).expanduser().resolve(),
                        analysis_result.to_dict(),
                        lyric=lyric,
                        track_result=offset_track_result,
                        alignment_result=offset_alignment_result,
                        selected_segment_indices=offset_segment_indices,
                        mode=args.align_mode,
                        fallback_to=offset_fallback_to,
                    )
                    print(f"已更新 Offset 调试 JSON: {args.offset_debug_json}")

            if offset_transcriber is not None:
                offset_transcriber.unload_model_if_needed()

            dp_transcriber = WhisperTranscriber(
                model_id=args.model_id,
                language=args.language,
                prompt_mode=args.prompt_mode,
                num_candidates=args.num_candidates,
                enable_token_timestamps=True,
                low_memory_mode=args.low_memory_mode,
            )
            track_result = dp_transcriber.transcribe_analysis(
                analysis_result=analysis_result,
                lyric_hint=lyric,
            )
            print("DP 分片转写:")
            print(f"  chunks={len(track_result.chunks)}")
            print(f"  stats={track_result.stats}")

            alignment_result = aligner.align(lyric=lyric, transcription=track_result)
            if args.align_mode == "auto":
                alignment_result.details = dict(alignment_result.details)
                alignment_result.details["fallback_from"] = "offset"
            if lyric_has_real_timestamps:
                alignment_result = aligner.refine_with_lyric_timestamps(
                    lyric=lyric,
                    alignment_result=alignment_result,
                    transcription=track_result,
                )
            print("歌词对齐:")
            print(f"  stats={alignment_result.stats}")
            hybrid_details = (alignment_result.details or {}).get("hybrid_refinement") or {}
            if hybrid_details.get("enabled"):
                print("  hybrid_refinement=True")
                print(f"  hybrid_anchor_count={hybrid_details.get('anchor_count')}")
                print(f"  hybrid_global_offset={hybrid_details.get('global_offset')}")
                section_offsets = hybrid_details.get("section_offsets") or {}
                if section_offsets:
                    print(f"  hybrid_section_offsets={section_offsets}")

        if track_result is None or alignment_result is None:
            raise RuntimeError("未能生成转写或对齐结果")

        aligned_lrc = build_aligned_lrc(
            lyric=lyric,
            alignment_result=alignment_result,
            include_translation=not args.no_translation,
            skip_unmatched=args.skip_unmatched,
        )
        if not aligned_lrc:
            raise RuntimeError("未生成任何可写回的歌词内容")

        write_lyric_metadata(audio_path, aligned_lrc)
        print(f"已写回歌词标签: {audio_path}")

        output_lrc = Path(args.output_lrc).expanduser().resolve() if args.output_lrc else audio_path.with_suffix(".aligned.lrc")
        output_lrc.parent.mkdir(parents=True, exist_ok=True)
        output_lrc.write_text(aligned_lrc, encoding="utf-8")
        print(f"已保存对齐歌词: {output_lrc}")

        if args.transcription_json:
            save_transcription_json(Path(args.transcription_json).expanduser().resolve(), analysis_result.to_dict(), track_result)
            print(f"已保存转写 JSON: {args.transcription_json}")

        if args.alignment_json:
            save_alignment_json(Path(args.alignment_json).expanduser().resolve(), alignment_result)
            print(f"已保存对齐 JSON: {args.alignment_json}")
    finally:
        analyzer.release_model_if_needed()
        if offset_transcriber is not None:
            offset_transcriber.unload_model_if_needed()
        if dp_transcriber is not None:
            dp_transcriber.unload_model_if_needed()
        lyric_executor.shutdown(wait=False, cancel_futures=True)


def _run_batch_directory(args: argparse.Namespace, input_dir: Path) -> None:
    audio_files = _iter_audio_files_in_directory(input_dir)
    log_path, err_path = _batch_log_paths(input_dir)
    log_entries: list[str] = []
    err_entries: list[str] = []
    failure_count = 0

    if not audio_files:
        message = f"目录下未找到可处理的音频文件: {input_dir}"
        log_path.write_text("", encoding="utf-8")
        err_path.write_text(f"{message}\n", encoding="utf-8")
        raise FileNotFoundError(message)

    for audio_file in audio_files:
        per_file_args = copy.copy(args)
        per_file_args.audio_path = str(audio_file)
        output = io.StringIO()
        error_output = io.StringIO()
        tee_stdout = _TeeStream(sys.stdout, output)
        tee_stderr = _TeeStream(sys.stderr, error_output)
        try:
            print(f"\n=== {audio_file.name} ===")
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                process_audio_file(per_file_args, audio_file)
        except Exception:
            failure_count += 1
            captured = output.getvalue().strip()
            captured_err = error_output.getvalue().strip()
            traceback_text = traceback.format_exc().strip()
            if captured:
                log_entries.append(f"=== {audio_file.name} ===\n{captured}")
            error_parts = [f"=== {audio_file.name} ==="]
            if captured_err:
                error_parts.append(captured_err)
            if captured:
                error_parts.append(captured)
            error_parts.append(traceback_text)
            err_entries.append("\n".join(part for part in error_parts if part))
            continue

        captured = output.getvalue().strip()
        captured_err = error_output.getvalue().strip()
        if captured:
            log_entries.append(f"=== {audio_file.name} ===\n{captured}")
        if captured_err:
            err_entries.append(f"=== {audio_file.name} ===\n{captured_err}")

    log_path.write_text("\n\n".join(log_entries) + ("\n" if log_entries else ""), encoding="utf-8")
    err_path.write_text("\n\n".join(err_entries) + ("\n" if err_entries else ""), encoding="utf-8")

    print("批处理完成:")
    print(f"  input_dir={input_dir}")
    print(f"  files={len(audio_files)}")
    print(f"  log={log_path}")
    print(f"  err={err_path}")
    print(f"  failed={failure_count}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.audio_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_dir():
        if args.output_lrc or args.transcription_json or args.alignment_json or args.offset_debug_json:
            raise RuntimeError(
                "目录批处理模式下不支持 --output-lrc / --transcription-json / --alignment-json / --offset-debug-json；"
                "请使用默认按文件输出"
            )
        if args.lyric_path:
            raise RuntimeError("目录批处理模式下不支持统一的 --lyric-path；请改用在线歌词或内嵌歌词")
        _run_batch_directory(args, input_path)
        return

    process_audio_file(args, input_path)
