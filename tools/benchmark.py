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
    errors = start_errors + end_errors
    percentile = lambda value: None if not errors else sorted(errors)[round((len(errors) - 1) * value)]
    return {
        "line_count": len(reference.lines),
        "matched_coverage": complete / max(len(reference.lines), 1),
        "unresolved_lines": sum(item.start is None for item in alignment.lines),
        "endpoint_mae_seconds": None if not errors else sum(errors) / len(errors),
        "endpoint_p50_seconds": percentile(0.50),
        "endpoint_p90_seconds": percentile(0.90),
        "endpoint_within_0_25_seconds": None if not errors else sum(item <= 0.25 for item in errors) / len(errors),
        "endpoint_within_0_50_seconds": None if not errors else sum(item <= 0.50 for item in errors) / len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Lazulite 三首整曲 benchmark")
    parser.add_argument("--fixture-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--backend", choices=("qwen", "whisper"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=("dp", "offset", "hybrid", "auto"), default="dp")
    parser.add_argument("--without-separation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    backend = QwenBackend(QwenConfig(arguments.model_path, "Qwen/Qwen3-ASR", arguments.revision)) if arguments.backend == "qwen" else WhisperBackend(WhisperConfig(arguments.model_path, "openai/whisper-large-v3", arguments.revision))
    separator = None if arguments.without_separation else DemucsSeparator(DemucsConfig())
    records: list[dict[str, object]] = []
    try:
        for source in sorted(arguments.fixture_dir.glob("*.m4a")):
            ttml = source.with_suffix(".ttml")
            if not ttml.exists():
                continue
            reference = read_ttml_for_alignment(ttml)
            # DP 评估不向自动算法泄漏人工时间。
            input_document = LyricDocument(tuple(replace(line, source_start=None, source_end=None) for line in reference.lines), reference.language)
            audio = decode_audio(source)
            started = perf_counter()
            result = AlignmentPipeline(transcriber=backend, separator=separator).run(audio, input_document, config=PipelineConfig(mode=arguments.mode, language=reference.language))
            elapsed = perf_counter() - started
            records.append({
                "audio": source.name,
                "audio_source_id": audio.source_id,
                "duration_seconds": audio.duration,
                "elapsed_seconds": elapsed,
                "rtf": elapsed / audio.duration,
                "backend": asdict(result.transcription.backend) if result.transcription else asdict(backend.info),
                "metrics": _metrics(reference, result.alignment),
                "diagnostics": list(result.alignment.diagnostics),
            })
    finally:
        if separator is not None:
            separator.close()
        backend.close()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
