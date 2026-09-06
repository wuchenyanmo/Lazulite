"""对三首固定素材运行整曲自动行对齐并输出精简指标。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

from lazulite.audio.io import decode_audio
from lazulite.audio.separation import DemucsConfig, DemucsSeparator
from lazulite.lyrics.ttml import read_ttml_for_alignment
from lazulite.models.lyrics import LyricDocument
from lazulite.pipeline import AlignmentPipeline, PipelineConfig
from lazulite.transcription.backends import QwenBackend, QwenConfig, WhisperBackend, WhisperConfig


def _metric_summary(errors: list[float]) -> dict[str, float | None]:
    if not errors:
        return {"mae_seconds": None, "p50_seconds": None, "p90_seconds": None, "within_0_25_seconds": None, "within_0_50_seconds": None}
    ordered = sorted(errors)
    return {
        "mae_seconds": sum(errors) / len(errors),
        "p50_seconds": ordered[round((len(ordered) - 1) * 0.50)],
        "p90_seconds": ordered[round((len(ordered) - 1) * 0.90)],
        "within_0_25_seconds": sum(item <= 0.25 for item in errors) / len(errors),
        "within_0_50_seconds": sum(item <= 0.50 for item in errors) / len(errors),
    }


def _metrics(reference: LyricDocument, alignment) -> dict[str, object]:
    truth = {line.id: line for line in reference.lines}
    start_errors: list[float] = []
    end_errors: list[float] = []
    complete = 0
    for item in alignment.lines:
        line = truth[item.line_id]
        if item.start is not None and line.source_start is not None:
            start_errors.append(abs(item.start - line.source_start))
        if item.end is not None and line.source_end is not None:
            end_errors.append(abs(item.end - line.source_end))
        if item.start is not None and item.end is not None:
            complete += 1
    return {
        "line_count": len(reference.lines),
        "complete_coverage": complete / max(len(reference.lines), 1),
        "unresolved_lines": sum(item.start is None for item in alignment.lines),
        "start": _metric_summary(start_errors),
        "end": _metric_summary(end_errors),
    }


def _evaluation_document(reference: LyricDocument, *, mode: str, source_shift: float) -> LyricDocument:
    if mode == "dp":
        return LyricDocument(tuple(replace(line, source_start=None, source_end=None) for line in reference.lines), reference.language)
    return LyricDocument(tuple(
        replace(
            line,
            source_start=None if line.source_start is None else line.source_start - source_shift,
            source_end=None if line.source_end is None else line.source_end - source_shift,
        ) for line in reference.lines
    ), reference.language)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Lazulite 三首整曲 benchmark")
    parser.add_argument("--fixture-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--backend", choices=("qwen", "whisper"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=("dp", "offset", "hybrid", "auto"), default="dp")
    parser.add_argument("--source-shift", type=float, default=4.0, help="非 DP 模式注入的已知源时间偏移秒数")
    parser.add_argument("--without-separation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    model_started = perf_counter()
    backend = QwenBackend(QwenConfig(arguments.model_path, "Qwen/Qwen3-ASR", arguments.revision)) if arguments.backend == "qwen" else WhisperBackend(WhisperConfig(arguments.model_path, "openai/whisper-large-v3", arguments.revision))
    model_load_seconds = perf_counter() - model_started
    separator = None if arguments.without_separation else DemucsSeparator(DemucsConfig())
    records: list[dict[str, object]] = []
    try:
        for source in sorted(arguments.fixture_dir.glob("*.m4a")):
            ttml = source.with_suffix(".ttml")
            if not ttml.exists():
                continue
            reference = read_ttml_for_alignment(ttml)
            input_document = _evaluation_document(reference, mode=arguments.mode, source_shift=arguments.source_shift)
            audio = decode_audio(source)
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            events: list[tuple[str, str, float]] = []
            started = perf_counter()
            result = AlignmentPipeline(
                transcriber=backend,
                separator=separator,
                event_sink=lambda event: events.append((event.stage, event.kind, perf_counter())),
            ).run(audio, input_document, config=PipelineConfig(mode=arguments.mode, language=reference.language))
            elapsed = perf_counter() - started
            transcription_started = next((time for stage, kind, time in events if (stage, kind) == ("transcription", "started")), None)
            transcription_completed = next((time for stage, kind, time in events if (stage, kind) == ("transcription", "completed")), None)
            records.append({
                "audio": source.name,
                "audio_source_id": audio.source_id,
                "duration_seconds": audio.duration,
                "elapsed_seconds": elapsed,
                "rtf": elapsed / audio.duration,
                "hot_transcription_seconds": None if transcription_started is None or transcription_completed is None else transcription_completed - transcription_started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                "backend": asdict(result.transcription.backend) if result.transcription else asdict(backend.info),
                "effective_config": {
                    "pipeline": asdict(PipelineConfig(mode=arguments.mode, language=reference.language)),
                    "transcriber": backend.cache_identity,
                    "separator": {"kind": "none"} if separator is None else {
                        "config": asdict(separator.config),
                        "cache_identity": separator.cache_identity,
                    },
                },
                "metrics": _metrics(reference, result.alignment),
                "diagnostics": list(result.alignment.diagnostics),
                "lines": asdict(result.alignment)["lines"],
            })
    finally:
        if separator is not None:
            separator.close()
        backend.close()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({
        "model_load_seconds": model_load_seconds,
        "mode": arguments.mode,
        "source_shift": None if arguments.mode == "dp" else arguments.source_shift,
        "records": records,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
