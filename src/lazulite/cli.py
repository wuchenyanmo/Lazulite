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


def _add_backend_options(parser: argparse.ArgumentParser, *, forced: bool = False, required: bool = True) -> None:
    parser.add_argument("--backend", choices=("qwen",) if forced else ("whisper", "qwen"), required=required)
    parser.add_argument("--model-path", type=Path, required=required)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device-map", default="cuda:0", help="Qwen 使用的 Transformers device_map")
    if forced:
        return
    parser.add_argument("--device", default="cuda", help="Whisper 使用的 Torch 设备")
    parser.add_argument("--precision", choices=("float16", "float32"), help="默认 CPU float32，CUDA float16")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Whisper 生成上限")
    parser.add_argument("--top-k", type=int, default=5, help="每个生成 token 保留的候选数")
    parser.add_argument("--full-distributions", type=Path, help="按分片导出完整 logprob NPZ 的目录")


def build_parser() -> argparse.ArgumentParser:
    """构造不加载模型或网络来源的参数解析器。"""
    parser = argparse.ArgumentParser(prog="lazulite", description="歌词搜索、转写证据与行级对齐自动化引擎")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    japanese = commands.add_parser("align-japanese", help="独立日语全局词级对齐")
    source = japanese.add_mutually_exclusive_group(required=True)
    source.add_argument("--vocals", type=Path, help="完整人声音频")
    source.add_argument("--acoustic-input", type=Path, help="显式声学 NPZ")
    source.add_argument("--decode-input", type=Path, help="显式解码 JSON")
    source.add_argument("--audio", type=Path, help="原曲音频（自动 Demucs 准备）")
    target = japanese.add_mutually_exclusive_group(required=True)
    target.add_argument("--targets", type=Path, help="lazulite-japanese-request-v1 JSON")
    target.add_argument("--ttml", type=Path, help="TTML 正文与 Ruby 最小投影")
    japanese.add_argument("--without-completion", action="store_true", help="只输出固定 profile 原始路线")
    japanese.add_argument("--model-path", type=Path, help="本地 Wav2Vec2 CTC 模型目录")
    japanese.add_argument("--device", default="cpu")
    japanese.add_argument("--acoustic-output", type=Path)
    japanese.add_argument("--decode-output", type=Path)
    japanese.add_argument("--hint-mode", choices=("off", "auto"), default="auto")
    japanese.add_argument("--max-decode-memory-bytes", type=int, default=512*1024*1024)
    japanese.add_argument("--output", type=Path, required=True)

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
    source = align.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="原曲音频")
    source.add_argument("--vocals", type=Path, help="完整人声音频")
    align.add_argument("--transcriptions-input", type=Path)
    align.add_argument("--transcriptions-output", type=Path)
    align.add_argument("--evidence-policy", choices=("generic", "whisper-lexical-v1"), default="generic")
    align.add_argument("lyrics", type=Path)
    _add_backend_options(align, required=False)
    align.add_argument("--language")
    align.add_argument("--mode", choices=("dp", "offset", "hybrid", "auto"), default="auto")
    align.add_argument("--separator-model")
    align.add_argument("--separator-device")
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

    if arguments.backend is None or arguments.model_path is None:
        raise ValueError("转写计算需要 --backend 和 --model-path")
    distributions = arguments.full_distributions
    if distributions is not None:
        distributions.mkdir(parents=True, exist_ok=True)
    if arguments.backend == "whisper":
        return WhisperBackend(WhisperConfig(
            model_path=arguments.model_path,
            model_id=arguments.model_id,
            device=arguments.device,
            precision=arguments.precision or ("float32" if arguments.device=="cpu" else "float16"),
            top_k=arguments.top_k,
            max_new_tokens=arguments.max_new_tokens,
            distribution_output_dir=distributions,
        ))
    return QwenBackend(QwenConfig(
        model_path=arguments.model_path,
        model_id=arguments.model_id,
        device_map=arguments.device_map,
        top_k=arguments.top_k,
        distribution_output_dir=distributions,
    ))


def _run_japanese(arguments: argparse.Namespace) -> int:
    from .models.japanese import JapaneseAlignmentRequest
    from .alignment.japanese.run import compute_acoustics, decode_japanese, finalize_japanese, JapaneseConfig
    from .exchange import read_acoustic, write_acoustic, read_decode, write_decode
    if arguments.targets is not None:
        request=JapaneseAlignmentRequest.from_dict(json.loads(arguments.targets.read_text("utf-8")))
    else:
        from .lyrics.ttml import read_ttml_for_japanese
        request=read_ttml_for_japanese(arguments.ttml)
    config=JapaneseConfig(hint_mode=arguments.hint_mode,max_decode_memory_bytes=arguments.max_decode_memory_bytes,
                          completion=not arguments.without_completion)
    if arguments.decode_input is not None:
        if arguments.model_path is not None or arguments.acoustic_output is not None:
            raise ValueError('解码输入不接受声学模型或声学输出')
        decoded=read_decode(arguments.decode_input)
        if decoded.request!=request: raise ValueError('解码目标与本次歌词不一致')
    else:
        if arguments.acoustic_input is not None:
            if arguments.model_path is not None: raise ValueError('声学输入不接受模型路径')
            acoustic=read_acoustic(arguments.acoustic_input)
        else:
            if arguments.model_path is None: raise ValueError('音频输入需要 --model-path')
            from .audio.preparation import prepare_vocals,prepare_existing_vocals
            from .audio.io import decode_native_audio,decode_audio
            if arguments.vocals is not None:
                prepared=prepare_existing_vocals(decode_native_audio(arguments.vocals))
            else:
                from .audio.separation import DemucsSeparator,DemucsConfig
                separator=DemucsSeparator(DemucsConfig(device=arguments.device))
                try: prepared=prepare_vocals(decode_audio(arguments.audio,sample_rate=44100,channels=2),separator=separator)
                finally: separator.close()
            from .alignment.japanese.backend import NextFireBackend,NextFireConfig
            backend=NextFireBackend(NextFireConfig(arguments.model_path,arguments.device))
            try: acoustic=compute_acoustics(prepared,backend=backend)
            finally: backend.close()
        if arguments.acoustic_output is not None: write_acoustic(arguments.acoustic_output,acoustic)
        decoded=decode_japanese(acoustic,request,config=config)
    if arguments.decode_output is not None: write_decode(arguments.decode_output,decoded)
    result=finalize_japanese(decoded,completion=config.completion)
    _write_json(arguments.output,asdict(result))
    return 0


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
        if backend is not None: backend.close()
    return 0


def _run_align(arguments: argparse.Namespace) -> int:
    from .audio.io import decode_audio
    from .audio.preparation import prepare_existing_vocals
    from .audio.io import decode_native_audio
    from .lyrics.ttml import read_ttml_for_alignment
    from .pipeline import AlignmentPipeline, PipelineConfig
    from .alignment.line import LineAlignmentConfig
    from .alignment.offset import OffsetAlignmentConfig
    from .transcription.run import TranscriptionConfig
    from .transcription.evidence import EvidenceConfig
    from .exchange import read_transcriptions,write_transcriptions
    if arguments.vocals is not None and (arguments.separator_model is not None or arguments.separator_device is not None):
        raise ValueError("完整人声输入不接受分离参数")
    document=read_ttml_for_alignment(arguments.lyrics)
    prepared=prepare_existing_vocals(decode_native_audio(arguments.vocals)) if arguments.vocals is not None else None
    supplied=read_transcriptions(arguments.transcriptions_input) if getattr(arguments,'transcriptions_input',None) else None
    if supplied is not None and getattr(arguments,'backend',None) is None and getattr(arguments,'model_path',None) is not None:
        raise ValueError('单独模型路径未指定转写后端')
    backend=_build_transcriber(arguments) if supplied is None or getattr(arguments,'backend',None) is not None else None
    separator=None
    try:
        if prepared is None:
            from .audio.separation import DemucsConfig,DemucsSeparator
            separator=DemucsSeparator(DemucsConfig(arguments.separator_model or 'htdemucs',arguments.separator_device or 'cuda'))
        evidence=EvidenceConfig(arguments.evidence_policy)
        config=PipelineConfig(mode=arguments.mode,language=arguments.language,
            line=LineAlignmentConfig(evidence=evidence),offset=OffsetAlignmentConfig(evidence=evidence),
            transcription=TranscriptionConfig(evidence=evidence))
        pipeline=AlignmentPipeline(transcriber=backend,separator=separator)
        result=pipeline.run_prepared(prepared,document,config=config,transcriptions=supplied) if prepared is not None else pipeline.run(
            decode_audio(arguments.audio,sample_rate=44100,channels=2),document,config=config,transcriptions=supplied)
        if getattr(arguments,'transcriptions_output',None): write_transcriptions(arguments.transcriptions_output,result.transcriptions)
        _write_json(arguments.output,{**asdict(result.alignment),'status':result.alignment.status,
            'coverage':result.alignment.coverage,'issue_counts':result.alignment.issue_counts})
    finally:
        if separator is not None:separator.close()
        if backend is not None: backend.close()
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
    ("align-japanese", None): _run_japanese,
    ("lyrics", "search"): _run_lyrics_search,
    ("transcribe", None): _run_transcribe,
    ("align", None): _run_align,
    ("forced-align", None): _run_forced_alignment,
}


def main(argv: list[str] | None = None) -> int:
    """运行命令；库异常保留原因，CLI 仅将其转为简洁的非零退出。"""
    arguments = build_parser().parse_args(argv)
    try:
        inputs=[getattr(arguments,name,None) for name in ('audio','vocals','targets','ttml','lyrics','text','acoustic_input','decode_input','transcriptions_input')]
        outputs=[getattr(arguments,name,None) for name in ('output','acoustic_output','decode_output','transcriptions_output')]
        paths=[p.resolve() for p in outputs if p is not None]
        model=getattr(arguments,'model_path',None)
        if len(paths)!=len(set(paths)) or any(p.resolve() in paths for p in inputs if p is not None) or (model is not None and any(p==model.resolve() or model.resolve() in p.parents for p in paths)):
            raise ValueError("输出不能覆盖输入、模型或其他输出")
        return _HANDLERS[(arguments.command, getattr(arguments, "lyrics_command", None))](arguments)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"lazulite: {error}", file=sys.stderr)
        return 2
