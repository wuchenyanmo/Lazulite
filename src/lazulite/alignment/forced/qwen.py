from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from lazulite.models.alignment import ForcedAlignmentRequest, ForcedAlignmentResult, ForcedAlignmentUnit
from lazulite.models.audio import AudioBuffer
from lazulite.models.transcription import BackendInfo


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentConfig:
    model_path: Path
    model_id: str
    revision: str
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
        self.info = BackendInfo("qwen-forced", config.model_id, config.revision, f"qwen-asr {version('qwen-asr')}; torch {torch.__version__}", config.precision)

    def align(self, audio: AudioBuffer, *, request: ForcedAlignmentRequest) -> ForcedAlignmentResult:
        if audio.sample_rate != 16_000: raise ValueError("Qwen forced aligner 当前要求输入为 16 kHz")
        waveform = audio.samples.mean(axis=0, dtype=np.float32)
        language = "Japanese" if request.language == "ja" else request.language
        with self._torch.inference_mode():
            output = self._model.align(audio=(waveform, 16_000), text=request.text, language=language)
        cursor = 0; units = []; diagnostics = []
        for raw in output[0]:
            text = str(raw.text)
            start = request.text.find(text, cursor)
            if start < 0:
                diagnostics.append(f"unmapped:{text}"); continue
            cursor = start + len(text)
            units.append(ForcedAlignmentUnit(request.line_id, (request.input_range[0] + start, request.input_range[0] + cursor), text, float(raw.start_time) + request.audio_origin, float(raw.end_time) + request.audio_origin, "word", None))
        return ForcedAlignmentResult(self.info, tuple(units), tuple(diagnostics))

    def close(self) -> None:
        del self._model
        if self._torch.cuda.is_available(): self._torch.cuda.empty_cache()
