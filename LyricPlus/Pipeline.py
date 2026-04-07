from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError, TXXX, USLT
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

from LyricPlus import LyricAligner, VocalAnalyzer, WhisperTranscriber
from LyricPlus.Debug import load_alignment_lyric, save_alignment_json, save_transcription_json
from LyricPlus.Lyric import LyricLineStamp, LyricTokenLine
from LyricPlus.Search.GetLyric import get_163_lyric
from LyricPlus.Search.SearchMusic import search_163_music


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
    song_id: str | int | None = None,
    min_search_score: float = 55.0,
) -> tuple[LyricLineStamp, dict[str, Any]]:
    if song_id is not None:
        lyric = get_163_lyric(song_id)
        if lyric is None:
            raise RuntimeError(f"未能获取 song_id={song_id} 对应歌词")
        return lyric, {"id": song_id, "match sore": None}

    if not metadata.title:
        raise ValueError("音频元数据中缺少标题，无法自动搜索歌词；请传入 --title 或 --song-id")

    candidates = search_163_music(
        name=metadata.title,
        duration=metadata.duration,
        artist=metadata.artist,
        album=metadata.album,
    )
    if not candidates:
        raise RuntimeError("歌词搜索没有返回任何候选结果")

    best = candidates[0]
    best_score = float(best.get("match sore", 0.0))
    if best_score < min_search_score:
        raise RuntimeError(
            f"搜索命中分数过低: {best_score:.1f} < {min_search_score:.1f}，"
            "请改用 --song-id 或手动提供 --lyric-path"
        )

    lyric = get_163_lyric(best["id"])
    if lyric is None:
        raise RuntimeError(f"已找到歌曲 id={best['id']}，但未能获取歌词")
    return lyric, best


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
        if item.start is not None
    ]
    if not offsets:
        return 0.0
    return sum(offsets) / len(offsets)


def build_aligned_lrc(
    lyric: LyricLineStamp,
    alignment_result,
    include_translation: bool = True,
    translation_brackets: str = "【】",
    prefer_original_on_unmatched: bool = True,
    skip_unmatched: bool = False,
) -> str:
    lines: list[str] = []
    average_offset = _estimate_average_alignment_offset(alignment_result)

    for key in lyric.metadata_keys:
        value = lyric.metadata[key]
        lines.append(f"[{key}:{value}]")

    for item in alignment_result.items:
        timestamp = item.start
        if timestamp is None and prefer_original_on_unmatched:
            timestamp = max(0.0, float(item.line.timestamp + average_offset))
        if timestamp is None:
            if skip_unmatched:
                continue
            continue
        text = _format_lyric_line_text(item.line, include_translation, translation_brackets)
        lines.append(f"{_format_lrc_timestamp(timestamp)}{text}")

    return "\n".join(lines).strip()


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取音频元数据，搜索歌词，执行分片转写与约束对齐，并将对齐歌词写回音频标签。"
    )
    parser.add_argument("audio_path", help="输入音频文件路径")
    parser.add_argument("--song-id", help="直接指定网易云歌曲 ID，跳过搜索")
    parser.add_argument("--title", help="覆盖音频标题元数据")
    parser.add_argument("--artist", help="覆盖音频歌手元数据")
    parser.add_argument("--album", help="覆盖音频专辑元数据")
    parser.add_argument("--lyric-path", help="本地歌词文件路径；提供后优先使用本地歌词而不是网络搜索")
    parser.add_argument("--language", help="Whisper 语言参数，例如 zh / ja / en")
    parser.add_argument("--model-id", default="model/whisper-large-v3", help="Whisper 模型目录或模型名")
    parser.add_argument("--prompt-mode", default="hybrid", choices=["none", "previous", "hint", "hybrid"])
    parser.add_argument("--num-candidates", type=int, default=1, help="每个分片重复转写次数")
    parser.add_argument("--min-search-score", type=float, default=55.0, help="歌词搜索最低接受分数")
    parser.add_argument("--output-lrc", help="将最终对齐歌词额外保存为 .lrc 文件")
    parser.add_argument("--transcription-json", help="将分片转写结果额外保存为 JSON")
    parser.add_argument("--alignment-json", help="将对齐结果额外保存为 JSON")
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audio_path = Path(args.audio_path).expanduser().resolve()
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

    if args.lyric_path:
        lyric = load_alignment_lyric(args.lyric_path, music_path=str(audio_path))
        if lyric is None:
            raise RuntimeError(f"无法从本地歌词文件加载歌词: {args.lyric_path}")
        search_result = {"source": "local", "id": None, "match sore": None}
    else:
        lyric, search_result = search_lyric_from_metadata(
            metadata=metadata,
            song_id=args.song_id,
            min_search_score=args.min_search_score,
        )

    print("歌词来源:")
    print(f"  source={search_result.get('source', 'netease')}")
    print(f"  song_id={search_result.get('id')}")
    print(f"  match_score={search_result.get('match sore')}")
    print(f"  lyric_lines={len(lyric.lyric_lines)}")

    analyzer = VocalAnalyzer(low_memory_mode=args.low_memory_mode)
    transcriber = WhisperTranscriber(
        model_id=args.model_id,
        language=args.language,
        prompt_mode=args.prompt_mode,
        num_candidates=args.num_candidates,
        low_memory_mode=args.low_memory_mode,
    )
    aligner = LyricAligner()

    try:
        analysis_result = analyzer.analyze_file(str(audio_path))
        print("人声分析:")
        print(f"  is_vocal={analysis_result.is_vocal}")
        print(f"  vocal_time={analysis_result.vocal_time:.2f}s")
        print(f"  segments={len(analysis_result.segments)}")

        analyzer.release_model_if_needed()

        track_result = transcriber.transcribe_analysis(
            analysis_result=analysis_result,
            lyric_hint=lyric,
        )
        print("分片转写:")
        print(f"  chunks={len(track_result.chunks)}")
        print(f"  stats={track_result.stats}")

        alignment_result = aligner.align(lyric=lyric, transcription=track_result)
        print("歌词对齐:")
        print(f"  stats={alignment_result.stats}")

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
        transcriber.unload_model_if_needed()
