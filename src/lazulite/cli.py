"""命令行边界。

此模块刻意不导入音频分离或转写后端；具体命令在相应能力完成后再装配。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """构造不触发模型加载的基础命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="lazulite",
        description="歌词搜索、转写证据与行级对齐自动化引擎",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="转写整段音频并输出机器证据 JSON")
    transcribe.add_argument("audio", type=Path); transcribe.add_argument("--backend", choices=("whisper", "qwen"), required=True)
    transcribe.add_argument("--model-path", type=Path, required=True); transcribe.add_argument("--language"); transcribe.add_argument("--output", type=Path, required=True)
    align = commands.add_parser("align", help="将 TTML 歌词行与音频对齐")
    align.add_argument("audio", type=Path); align.add_argument("lyrics", type=Path); align.add_argument("--backend", choices=("whisper", "qwen"), required=True)
    align.add_argument("--model-path", type=Path, required=True); align.add_argument("--language"); align.add_argument("--mode", choices=("dp", "offset", "hybrid", "auto"), default="auto"); align.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行轻量 CLI；后续子命令在此边界安装运行资源。"""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "transcribe":
        from .audio.io import decode_audio
        from .models.audio import VocalSegment
        from .transcription.base import TranscriptionRequest
        from .transcription.run import transcribe_segments
        from .transcription.backends import QwenBackend, QwenConfig, WhisperBackend, WhisperConfig
        audio = decode_audio(arguments.audio)
        segment = VocalSegment("full-audio", None, 0, audio.duration, 0, audio.duration, None, None)
        backend = WhisperBackend(WhisperConfig(arguments.model_path, "openai/whisper-large-v3", "local")) if arguments.backend == "whisper" else QwenBackend(QwenConfig(arguments.model_path, "Qwen/Qwen3-ASR", "local"))
        result = transcribe_segments(audio, (segment,), transcriber=backend, request=TranscriptionRequest(language=arguments.language, include_timing=arguments.backend == "whisper"))
        arguments.output.write_text(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=2))
        backend.close()
    elif arguments.command == "align":
        from .audio.io import decode_audio
        from .lyrics.ttml import read_ttml_for_alignment
        from .pipeline import AlignmentPipeline, PipelineConfig
        from .transcription.backends import QwenBackend, QwenConfig, WhisperBackend, WhisperConfig
        backend = WhisperBackend(WhisperConfig(arguments.model_path, "openai/whisper-large-v3", "local")) if arguments.backend == "whisper" else QwenBackend(QwenConfig(arguments.model_path, "Qwen/Qwen3-ASR", "local"))
        result = AlignmentPipeline(transcriber=backend).run(decode_audio(arguments.audio), read_ttml_for_alignment(arguments.lyrics), config=PipelineConfig(mode=arguments.mode, language=arguments.language))
        arguments.output.write_text(json.dumps(asdict(result.alignment), ensure_ascii=False, allow_nan=False, indent=2))
        backend.close()
    return 0
