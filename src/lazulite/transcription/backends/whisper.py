"""Transformers Whisper generate 适配器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import numpy as np

from lazulite.models.transcription import BackendInfo, TimedText, TranscriptionSegment, ProbabilitySummary
from lazulite.transcription.base import TranscriberCapabilities, TranscriptionInput, TranscriptionRequest
from lazulite.transcription.token_scores import export_full_logprobs, extract_token_scores


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    model_path: Path
    model_id: str | None = None
    device: str = "cuda"
    precision: str = "float16"
    attention_implementation: str = "eager"
    max_new_tokens: int = 128
    top_k: int = 5
    distribution_output_dir: Path | None = None

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


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
    capabilities = TranscriberCapabilities("subword", context=True, max_audio_seconds=30.0)

    @property
    def config(self):
        return self._config

    def __init__(self, config: WhisperConfig) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Whisper 后端需要安装 lazulite[whisper] 与匹配的 CUDA Torch") from error
        self._torch = torch
        self._config = config
        dtype = getattr(torch, config.precision)
        from lazulite._runtime import preserve_random_state
        with preserve_random_state(torch):
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(str(config.model_path), dtype=dtype, attn_implementation=config.attention_implementation).to(config.device).eval()
            self._processor = AutoProcessor.from_pretrained(str(config.model_path))
        self.info = BackendInfo("whisper", config.model_id, f"transformers {version('transformers')}; torch {torch.__version__}", config.precision)
        self._export_index = 0

    def _waveform(self, item: TranscriptionInput) -> np.ndarray:
        if item.audio.sample_rate != 16_000:
            raise ValueError("WhisperBackend 当前要求输入已解码为 16 kHz")
        start = int(round(item.segment.start * item.audio.sample_rate))
        end = int(round(item.segment.end * item.audio.sample_rate))
        return item.audio.samples[:, start:end].mean(axis=0, dtype=np.float32)

    def _timed_text(self, ids: list[int], prefix: int, timestamps, item: TranscriptionInput) -> tuple[TimedText, ...]:
        values = timestamps[0].detach().float().cpu().tolist() if timestamps is not None else []
        special = set(self._processor.tokenizer.all_special_ids)
        pending = bytearray(); group_start: int | None = None; groups: list[tuple[str, int, int]] = []
        for step, token_id in enumerate(ids):
            if token_id in special:
                continue
            token = self._processor.tokenizer.convert_ids_to_tokens(token_id)
            try:
                if group_start is None:
                    group_start = step
                pending.extend(_BYTE_DECODER[character] for character in token)
                text = pending.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if text:
                # 一个 Unicode 字符可能跨多个 BPE token；开始边界必须取首 token。
                groups.append((text, group_start, step + 1))
                pending.clear(); group_start = None
        result: list[TimedText] = []
        for index, (text, first_step, last_step) in enumerate(groups):
            sequence_index = prefix + first_step
            start = values[sequence_index] if sequence_index < len(values) else None
            next_step = groups[index + 1][1] if index + 1 < len(groups) else None
            end = values[prefix + next_step] if next_step is not None and prefix + next_step < len(values) else None
            if start is not None:
                start += item.segment.start
            if end is not None:
                end += item.segment.start
            if start is not None and end is not None and end <= start:
                end = None
            result.append(TimedText(text, start, end, "subword", "model", "model_timestamp" if start is not None else None, "successor_start" if end is not None else None, tuple(range(first_step,last_step))))
        return tuple(result)

    def _one(self, item: TranscriptionInput, request: TranscriptionRequest) -> TranscriptionSegment:
        inputs = self._processor(self._waveform(item), sampling_rate=16_000, return_tensors="pt", return_attention_mask=True).to(self._config.device)
        inputs["input_features"] = inputs["input_features"].to(getattr(self._torch, self._config.precision))
        if request.context and request.include_timing:
            raise ValueError("Whisper prompt is unsupported with token timing")
        prompt_kwargs = {}
        if request.context:
            prompt_kwargs["prompt_ids"] = self._processor.get_prompt_ids(request.context, return_tensors="pt").to(self._config.device)
        from lazulite._runtime import preserve_random_state
        with preserve_random_state(self._torch), self._torch.inference_mode():
            output = self._model.generate(**inputs, **prompt_kwargs, language=request.language, task="transcribe", max_new_tokens=self._config.max_new_tokens, do_sample=False, return_token_timestamps=request.include_timing, return_dict_in_generate=True, output_scores=True, condition_on_prev_tokens=False)
        generation_scores = output["scores"]
        if not generation_scores:
            raise RuntimeError("Whisper generate 未返回 scores，不能伪造 token 概率")
        sequence = output["sequences"][0].tolist(); prefix = len(sequence) - len(generation_scores)
        ids = sequence[prefix:]
        logits = np.stack([score[0].float().cpu().numpy() for score in generation_scores])
        special = set(self._processor.tokenizer.all_special_ids)
        lexical_steps = [step for step, token_id in enumerate(ids) if token_id not in special]
        scores = extract_token_scores(logits, ids, distribution_kind="generation_processed", lexical_steps=lexical_steps, top_k=self._config.top_k)
        metadata = scores.metadata(); metadata.update({"generated_ids": ids, "prefix_length": prefix, "body_generation_steps": lexical_steps, "request_context": request.context})
        if self._config.distribution_output_dir is not None:
            target = self._config.distribution_output_dir / f"{item.segment.id}-{self._export_index}.npz"; self._export_index += 1
            metadata["distribution_file"] = str(export_full_logprobs(target, scores, ids, lexical_steps=lexical_steps, source_segment_id=item.segment.id, model_id=self.info.model_id))
        timestamps = output.get("token_timestamps")
        eos = getattr(self._processor.tokenizer, 'eos_token_id', None)
        stop = 'eos' if eos is not None and eos in ids else 'max_new_tokens' if len(ids)>=self._config.max_new_tokens else 'unknown'
        raw_times = timestamps[0].detach().float().cpu().tolist() if timestamps is not None else None
        metadata.update({'sequence_ids':sequence,'prefix_ids':sequence[:prefix],
                         'raw_token_timestamps':raw_times,'stop_reason':stop,
                         'generation_parameters':{'max_new_tokens':self._config.max_new_tokens,'do_sample':False,
                             'condition_on_prev_tokens':False,'task':'transcribe','language':request.language,
                             'return_token_timestamps':request.include_timing},
                         'eos_timestamp_copied': bool(raw_times is not None and eos is not None and eos in ids
                             and prefix+ids.index(eos)<len(raw_times) and prefix+ids.index(eos)>0
                             and raw_times[prefix+ids.index(eos)]==raw_times[prefix+ids.index(eos)-1])})
        probabilities={x.generation_step:float(np.exp(x.logprob)) for x in scores.scores}
        def summarize(steps):
            values=[probabilities[i] for i in steps if i in probabilities]
            return ProbabilitySummary(float(np.mean(values)),min(values),sum(v<.35 for v in values)/len(values),len(values)) if values else None
        from dataclasses import replace
        units=tuple(replace(u,probability=summarize(u.generation_steps)) for u in self._timed_text(ids,prefix,timestamps,item))
        metadata['unicode_groups']=[{'text':u.text,'generation_steps':list(u.generation_steps)} for u in units]
        text = self._processor.batch_decode([ids], skip_special_tokens=True)[0]
        return TranscriptionSegment(item.segment.id, text, request.language, None, None, scores.confidence, "generation_token_geometric_mean" if scores.confidence is not None else None, units, metadata, probability=summarize(lexical_steps), issues=("generation_truncated",) if stop=="max_new_tokens" else ())

    def transcribe(self, inputs: Sequence[TranscriptionInput], *, request: TranscriptionRequest) -> tuple[TranscriptionSegment, ...]:
        return tuple(self._one(item, request) for item in inputs)

    def close(self) -> None:
        # 生命周期结束时断开模型引用，不在库内强制干预调用方的 CUDA allocator。
        self._model = None
        self._processor = None
