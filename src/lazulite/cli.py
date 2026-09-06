"""命令行边界：只解析参数、装配运行资源和写出调用方指定的 JSON。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Callable

from . import __version__


def _write_json(path: Path, value: object) -> None:
    """只写调用方指定的结果文件，绝不触碰输入媒体。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def _add_backend_options(parser: argparse.ArgumentParser, *, forced: bool = False) -> None:
    parser.add_argument("--backend", choices=("qwen",) if forced else ("whisper", "qwen"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ForcedAligner-0.6B" if forced else None)
    parser.add_argument("--revision", required=True, help="模型权重对应的固定 revision 或 SHA256")
    parser.add_argument("--device", default="cuda", help="Whisper/Demucs 使用的 Torch 设备")
    parser.add_argument("--device-map", default="cuda:0", help="Qwen 使用的 Transformers device_map")
    parser.add_argument("--top-k", type=int, default=5, help="每个生成 token 保留的候选数")
    parser.add_argument("--full-distributions", type=Path, help="按分片导出完整 logprob NPZ 的目录")


def build_parser() -> argparse.ArgumentParser:
    """构造不加载模型或网络来源的参数解析器。"""
    parser = argparse.ArgumentParser(prog="lazulite", description="歌词搜索、转写证据与行级对齐自动化引擎")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    lyrics = commands.add_parser("lyrics", help="从在线来源搜索歌词候选")
    lyric_commands = lyrics.add_subparsers(dest="lyrics_command", required=True)
    search = lyric_commands.add_parser("search", help="读取音频元数据并搜索歌词")
    search.add_argument("audio", type=Path)
    search.add_argument("--min-match-score", type=float, default=0.0)
    search.add_argument("--per-provider-limit", type=int, default=3)
    search.add_argument("--output", type=Path, required=True)

    transcribe = commands.add_parser("transcribe", help="转写整段音频并输出机器证据 JSON")
    transcribe.add_argument("audio", type=Path)
    _add_backend_options(transcribe)
    transcribe.add_argument("--language")
    transcribe.add_argument("--output", type=Path, required=True)

    align = commands.add_parser("align", help="将 TTML 歌词行与音频对齐")
    align.add_argument("audio", type=Path)
    align.add_argument("lyrics", type=Path)
    _add_backend_options(align)
    align.add_argument("--language")
    align.add_argument("--mode", choices=("dp", "offset", "hybrid", "auto"), default="auto")
    align.add_argument("--without-separation", action="store_true", help="跳过 Demucs，仅用于已有分离人声或诊断")
    align.add_argument("--separator-model", default="htdemucs")
    align.add_argument("--separator-device", default="cuda")
    align.add_argument("--output", type=Path, required=True)

    forced = commands.add_parser("forced-align", help="用已确认文本执行独立 forced alignment")
    forced.add_argument("audio", type=Path)
    forced.add_argument("text", type=Path, help="只包含已确认正文的 UTF-8 文本文件")
    _add_backend_options(forced, forced=True)
    forced.add_argument("--language")
    forced.add_argument("--line-id", default="line-0")
    forced.add_argument("--audio-origin", type=float, default=0.0, help="输入音频窗口相对于整曲的秒数")
    forced.add_argument("--output", type=Path, required=True)
    return parser


def _build_transcriber(arguments: argparse.Namespace):
    from .transcription.backends import QwenBackend, QwenConfig, WhisperBackend, WhisperConfig

    distributions = arguments.full_distributions
    if distributions is not None:
        distributions.mkdir(parents=True, exist_ok=True)
    if arguments.backend == "whisper":
        return WhisperBackend(WhisperConfig(
            model_path=arguments.model_path,
            model_id=arguments.model_id or "openai/whisper-large-v3",
            revision=arguments.revision,
            device=arguments.device,
            top_k=arguments.top_k,
            distribution_output_dir=distributions,
        ))
    return QwenBackend(QwenConfig(
        model_path=arguments.model_path,
        model_id=arguments.model_id or "Qwen/Qwen3-ASR",
        revision=arguments.revision,
        device_map=arguments.device_map,
        top_k=arguments.top_k,
        distribution_output_dir=distributions,
    ))


def _run_lyrics_search(arguments: argparse.Namespace) -> int:
    from .audio.io import read_audio_metadata
    from .lyrics.service import search_lyrics

    candidates = search_lyrics(
        read_audio_metadata(arguments.audio),
        min_match_score=arguments.min_match_score,
        per_provider_limit=arguments.per_provider_limit,
    )
    _write_json(arguments.output, [asdict(candidate) for candidate in candidates])
    return 0


def _run_transcribe(arguments: argparse.Namespace) -> int:
    from .audio.io import decode_audio
    from .models.audio import VocalSegment
    from .transcription.base import TranscriptionRequest
    from .transcription.run import transcribe_segments

    backend = _build_transcriber(arguments)
    try:
        audio = decode_audio(arguments.audio)
        segment = VocalSegment("full-audio", None, 0, audio.duration, 0, audio.duration, None, None)
        result = transcribe_segments(
            audio,
            (segment,),
            transcriber=backend,
            request=TranscriptionRequest(language=arguments.language, include_timing=arguments.backend == "whisper"),
        )
        _write_json(arguments.output, asdict(result))
    finally:
        backend.close()
    return 0


def _run_align(arguments: argparse.Namespace) -> int:
    from .audio.io import decode_audio
    from .lyrics.ttml import read_ttml_for_alignment
    from .pipeline import AlignmentPipeline, PipelineConfig

    backend = _build_transcriber(arguments)
    separator = None
    try:
        if not arguments.without_separation:
            from .audio.separation import DemucsConfig, DemucsSeparator
            separator = DemucsSeparator(DemucsConfig(arguments.separator_model, arguments.separator_device))
        result = AlignmentPipeline(transcriber=backend, separator=separator).run(
            decode_audio(arguments.audio),
            read_ttml_for_alignment(arguments.lyrics),
            config=PipelineConfig(mode=arguments.mode, language=arguments.language),
        )
        _write_json(arguments.output, asdict(result.alignment))
    finally:
        if separator is not None:
            separator.close()
        backend.close()
    return 0


def _run_forced_alignment(arguments: argparse.Namespace) -> int:
    from .alignment.forced.qwen import QwenForcedAligner, QwenForcedAlignmentConfig
    from .audio.io import decode_audio
    from .models.alignment import ForcedAlignmentRequest

    text = arguments.text.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("确认文本不能为空")
    aligner = QwenForcedAligner(QwenForcedAlignmentConfig(
        model_path=arguments.model_path,
        model_id=arguments.model_id,
        revision=arguments.revision,
        device_map=arguments.device_map,
    ))
    try:
        request = ForcedAlignmentRequest(arguments.line_id, text, arguments.language, arguments.audio_origin, (0, len(text)), "word")
        result = aligner.align(decode_audio(arguments.audio), request=request)
        _write_json(arguments.output, asdict(result))
    finally:
        aligner.close()
    return 0


_HANDLERS: dict[tuple[str, str | None], Callable[[argparse.Namespace], int]] = {
    ("lyrics", "search"): _run_lyrics_search,
    ("transcribe", None): _run_transcribe,
    ("align", None): _run_align,
    ("forced-align", None): _run_forced_alignment,
}


def main(argv: list[str] | None = None) -> int:
    """运行命令；库异常保留原因，CLI 仅将其转为简洁的非零退出。"""
    arguments = build_parser().parse_args(argv)
    try:
        return _HANDLERS[(arguments.command, getattr(arguments, "lyrics_command", None))](arguments)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"lazulite: {error}", file=sys.stderr)
        return 2
