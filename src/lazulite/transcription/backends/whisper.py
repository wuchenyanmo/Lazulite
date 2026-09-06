"""Transformers Whisper generate 适配器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import numpy as np

from lazulite.models.transcription import BackendInfo, TimedText, TranscriptionSegment
from lazulite.transcription.base import TranscriberCapabilities, TranscriptionInput, TranscriptionRequest
from lazulite.transcription.token_scores import export_full_logprobs, extract_token_scores


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    model_path: Path
    model_id: str
    revision: str
    device: str = "cuda"
    precision: str = "float16"
    attention_implementation: str = "eager"
    max_new_tokens: int = 128
    top_k: int = 5
    distribution_output_dir: Path | None = None


def _byte_decoder() -> dict[str, int]:
    initial = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    extended = initial.copy()
    extra = 0
    for byte in range(256):
        if byte not in initial:
            initial.append(byte)
            extended.append(256 + extra)
            extra += 1
    return dict(zip((chr(value) for value in extended), initial))


_BYTE_DECODER = _byte_decoder()


class WhisperBackend:
    capabilities = TranscriberCapabilities("subword", context=False, max_audio_seconds=30.0)

    def __init__(self, config: WhisperConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Whisper 后端需要安装 lazulite[whisper] 与匹配的 CUDA Torch") from error
        self._torch = torch
        self._config = config
        dtype = getattr(torch, config.precision)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(str(config.model_path), dtype=dtype, attn_implementation=config.attention_implementation).to(config.device).eval()
        self._processor = AutoProcessor.from_pretrained(str(config.model_path))
        self.info = BackendInfo("whisper", config.model_id, config.revision, f"transformers {version('transformers')}; torch {torch.__version__}", config.precision)
        self._export_index = 0

    @property
    def cache_identity(self) -> dict[str, object]:
        """缓存只依赖可审计的模型身份和实际生成参数，不依赖目录路径。"""
        config = asdict(self._config)
        config["model_path"] = None
        config["distribution_output_dir"] = str(self._config.distribution_output_dir) if self._config.distribution_output_dir else None
        return {"backend": asdict(self.info), "config": config}

    def _waveform(self, item: TranscriptionInput) -> np.ndarray:
        if item.audio.sample_rate != 16_000:
            raise ValueError("WhisperBackend 当前要求输入已解码为 16 kHz")
        start = int(round(item.segment.start * item.audio.sample_rate))
        end = int(round(item.segment.end * item.audio.sample_rate))
        return item.audio.samples[:, start:end].mean(axis=0, dtype=np.float32)

    def _timed_text(self, ids: list[int], prefix: int, timestamps, item: TranscriptionInput) -> tuple[TimedText, ...]:
        if timestamps is None:
            return ()
        values = timestamps[0].detach().float().cpu().tolist()
        special = set(self._processor.tokenizer.all_special_ids)
        pending = bytearray(); groups: list[tuple[str, int]] = []
        for step, token_id in enumerate(ids):
            if token_id in special:
                continue
            token = self._processor.tokenizer.convert_ids_to_tokens(token_id)
            try:
                pending.extend(_BYTE_DECODER[character] for character in token)
                text = pending.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if text:
                groups.append((text, step))
            pending.clear()
        result: list[TimedText] = []
        for index, (text, step) in enumerate(groups):
            sequence_index = prefix + step
            start = values[sequence_index] if sequence_index < len(values) else None
            next_step = groups[index + 1][1] if index + 1 < len(groups) else None
            end = values[prefix + next_step] if next_step is not None and prefix + next_step < len(values) else None
            if start is not None:
                start += item.segment.start
            if end is not None:
                end += item.segment.start
            if start is not None and end is not None and end <= start:
                end = None
            result.append(TimedText(text, start, end, "subword", "model"))
        return tuple(result)

    def _one(self, item: TranscriptionInput, request: TranscriptionRequest) -> TranscriptionSegment:
        inputs = self._processor(self._waveform(item), sampling_rate=16_000, return_tensors="pt", return_attention_mask=True).to(self._config.device)
        inputs["input_features"] = inputs["input_features"].to(getattr(self._torch, self._config.precision))
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, language=request.language, task="transcribe", max_new_tokens=self._config.max_new_tokens, do_sample=False, return_token_timestamps=request.include_timing, return_dict_in_generate=True, output_scores=True, condition_on_prev_tokens=False)
        generation_scores = output["scores"]
        if not generation_scores:
            raise RuntimeError("Whisper generate 未返回 scores，不能伪造 token 概率")
        sequence = output["sequences"][0].tolist(); prefix = len(sequence) - len(generation_scores)
        ids = sequence[prefix:]
        logits = np.stack([score[0].float().cpu().numpy() for score in generation_scores])
        special = set(self._processor.tokenizer.all_special_ids)
        lexical_steps = [step for step, token_id in enumerate(ids) if token_id not in special]
        scores = extract_token_scores(logits, ids, distribution_kind="generation_processed", lexical_steps=lexical_steps, top_k=self._config.top_k)
        metadata = scores.metadata(); metadata.update({"generated_ids": ids, "prefix_length": prefix, "body_generation_steps": lexical_steps})
        if self._config.distribution_output_dir is not None:
            target = self._config.distribution_output_dir / f"{item.segment.id}-{self._export_index}.npz"; self._export_index += 1
            metadata["distribution_file"] = str(export_full_logprobs(target, scores, ids, lexical_steps=lexical_steps, source_segment_id=item.segment.id, model_id=self.info.model_id, tokenizer_revision=self.info.revision))
        timestamps = output.get("token_timestamps")
        text = self._processor.batch_decode(output["sequences"], skip_special_tokens=True)[0]
        return TranscriptionSegment(item.segment.id, text, request.language, None, None, scores.confidence, "generation_token_geometric_mean" if scores.confidence is not None else None, self._timed_text(ids, prefix, timestamps, item), metadata)

    def transcribe(self, inputs: Sequence[TranscriptionInput], *, request: TranscriptionRequest) -> tuple[TranscriptionSegment, ...]:
        return tuple(self._one(item, request) for item in inputs)

    def close(self) -> None:
        del self._model
        if self._torch.cuda.is_available(): self._torch.cuda.empty_cache()
