from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import numpy as np

from lazulite.models.alignment import ForcedAlignmentRequest, ForcedAlignmentResult, ForcedAlignmentUnit
from lazulite.models.audio import AudioBuffer
from lazulite.models.transcription import BackendInfo


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentConfig:
    model_path: Path
    model_id: str | None = None
    device_map: str = "cuda:0"
    precision: str = "bfloat16"
    attention_implementation: str = "sdpa"


class QwenForcedAligner:
    """实验性 Qwen forced aligner；单位粒度以实际输出为准。"""
    def __init__(self, config: QwenForcedAlignmentConfig) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as error:
            raise RuntimeError("Qwen forced alignment 需要 lazulite[forced-qwen]") from error
        self._torch = torch
        self._model = Qwen3ForcedAligner.from_pretrained(str(config.model_path), dtype=getattr(torch, config.precision), device_map=config.device_map, attn_implementation=config.attention_implementation)
        self.info = BackendInfo("qwen-forced", config.model_id, f"qwen-asr {version('qwen-asr')}; torch {torch.__version__}", config.precision)

    @staticmethod
    def _granularity(text: str) -> str:
        """Qwen 的公开结果在 CJK 为单字符、其他语言常为词，按实际单位标记。"""
        return "character" if len(text) == 1 else "word"

    @staticmethod
    def _map_units(raw_units: Iterable[object], request: ForcedAlignmentRequest) -> tuple[tuple[ForcedAlignmentUnit, ...], tuple[str, ...]]:
        """按输出顺序映射，重复文本不回跳；不明确的单位绝不猜测范围。"""
        range_start, range_end = request.input_range
        input_text = request.text[range_start:range_end]
        cursor = 0
        units: list[ForcedAlignmentUnit] = []
        diagnostics: list[str] = []
        for raw in raw_units:
            text = str(getattr(raw, "text", ""))
            if not text:
                diagnostics.append("unmapped:empty_unit")
                continue
            start = input_text.find(text, cursor)
            if start < 0:
                diagnostics.append(f"unmapped:{text}")
                continue
            end = start + len(text)
            cursor = end
            try:
                start_time = float(getattr(raw, "start_time")) + request.audio_origin
                end_time = float(getattr(raw, "end_time")) + request.audio_origin
                units.append(ForcedAlignmentUnit(
                    request.line_id,
                    (range_start + start, range_start + end),
                    text,
                    start_time,
                    end_time,
                    QwenForcedAligner._granularity(text),
                    None,
                ))
            except (AttributeError, TypeError, ValueError):
                diagnostics.append(f"invalid_timing:{text}")
        return tuple(units), tuple(diagnostics)

    def align(self, audio: AudioBuffer, *, request: ForcedAlignmentRequest) -> ForcedAlignmentResult:
        if audio.sample_rate != 16_000: raise ValueError("Qwen forced aligner 当前要求输入为 16 kHz")
        if request.reading_spans:
            raise ValueError("Qwen forced aligner 当前不支持将读音范围映射回原文；请传入已确认的单一文本范围")
        waveform = audio.samples.mean(axis=0, dtype=np.float32)
        language = "Japanese" if request.language == "ja" else request.language
        with self._torch.inference_mode():
            output = self._model.align(
                audio=(waveform, 16_000),
                text=request.text[request.input_range[0]:request.input_range[1]],
                language=language,
            )
        if len(output) != 1:
            raise RuntimeError("Qwen forced aligner 未返回单个输入对应的结果")
        units, diagnostics = self._map_units(output[0].items, request)
        return ForcedAlignmentResult(self.info, units, diagnostics)

    def close(self) -> None:
        self._model = None
