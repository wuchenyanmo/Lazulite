"""官方 qwen-asr 底层 generate 适配器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from lazulite.models.transcription import BackendInfo, TranscriptionSegment
from lazulite.transcription.base import (
    TranscriberCapabilities,
    TranscriptionInput,
    TranscriptionRequest,
)
from lazulite.transcription.token_scores import export_full_logprobs, extract_token_scores


@dataclass(frozen=True, slots=True)
class QwenConfig:
    model_path: Path
    model_id: str | None = None
    device_map: str = "cuda:0"
    precision: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 128
    top_k: int = 5
    distribution_output_dir: Path | None = None


class QwenBackend:
    """Qwen3-ASR 的无时间转写后端；概率不被误写成 token 时间。"""

    capabilities = TranscriberCapabilities("none", context=False, max_audio_seconds=None)

    @property
    def config(self):
        return self._config

    def __init__(self, config: QwenConfig) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel, parse_asr_output
        except ImportError as error:
            raise RuntimeError("Qwen 后端需要安装 lazulite[qwen] 与匹配的 CUDA Torch") from error
        self._torch = torch
        self._parse_asr_output = parse_asr_output
        self._config = config
        dtype = getattr(torch, config.precision)
        self._wrapper = Qwen3ASRModel.from_pretrained(
            str(config.model_path),
            dtype=dtype,
            device_map=config.device_map,
            max_inference_batch_size=1,
            max_new_tokens=config.max_new_tokens,
            attn_implementation=config.attention_implementation,
        )
        self.info = BackendInfo(
            name="qwen-asr",
            model_id=config.model_id,
            runtime_version=f"qwen-asr {version('qwen-asr')}; transformers {version('transformers')}; torch {torch.__version__}",
            precision=config.precision,
        )
        self._export_index = 0

    @staticmethod
    def _audio_for_input(item: TranscriptionInput) -> np.ndarray:
        audio = item.audio
        if audio.sample_rate != 16_000:
            raise ValueError("QwenBackend 当前要求输入已解码为 16 kHz")
        start = int(round(item.segment.start * audio.sample_rate))
        end = int(round(item.segment.end * audio.sample_rate))
        return audio.samples[:, start:end].mean(axis=0, dtype=np.float32)

    @staticmethod
    def _body_output(
        tokenizer: object,
        ids: list[int],
        *,
        forced_language: str | None,
        parse_asr_output: Callable[[str, str | None], tuple[str, str]],
    ) -> tuple[str | None, str, list[int]]:
        """从自动语言头中剥离正文，并返回与正文严格对应的生成步。"""
        decode = lambda values: tokenizer.batch_decode(  # type: ignore[attr-defined]
            [values], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        raw_text = decode(ids)
        language, text = parse_asr_output(raw_text, user_language=forced_language)
        special_ids = set(tokenizer.all_special_ids)  # type: ignore[attr-defined]
        if forced_language is not None or "<asr_text>" not in raw_text:
            start = 0
        else:
            start = len(ids)
            # 使用前缀逐步解码定位标签结束，不能按 Unicode 字符位置切 token ID。
            for index in range(1, len(ids) + 1):
                if "<asr_text>" in decode(ids[:index]):
                    start = index
                    break
        return language or None, text, [index for index, token_id in enumerate(ids) if index >= start and token_id not in special_ids]

    def _one(self, item: TranscriptionInput, request: TranscriptionRequest) -> TranscriptionSegment:
        language = "Japanese" if request.language == "ja" else request.language
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        prompt = self._wrapper.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            prompt += f"language {language}<asr_text>"
        inputs = self._wrapper.processor(
            text=[prompt], audio=[self._audio_for_input(item)],
            return_tensors="pt", padding=True,
        ).to(self._wrapper.model.device).to(self._wrapper.model.dtype)
        with self._torch.inference_mode():
            output = self._wrapper.model.generate(
                **inputs, max_new_tokens=self._config.max_new_tokens, do_sample=False,
                output_logits=True,
            )
        if output.logits is None or len(output.logits) == 0:
            raise RuntimeError("Qwen generate 未返回 raw logits，不能伪造 token 概率")
        prompt_length = int(inputs.input_ids.shape[1])
        generated = output.sequences[0, prompt_length:]
        if len(generated) != len(output.logits):
            raise RuntimeError("Qwen 生成 token 数与 logits 步数不一致")
        ids = generated.tolist()
        logits = np.stack([step[0].float().cpu().numpy() for step in output.logits])
        detected_language, text, lexical_steps = self._body_output(
            self._wrapper.processor.tokenizer,
            ids,
            forced_language=language,
            parse_asr_output=self._parse_asr_output,
        )
        scores = extract_token_scores(logits, ids, distribution_kind="raw_model", lexical_steps=lexical_steps, top_k=self._config.top_k)
        metadata = scores.metadata()
        metadata["generated_ids"] = ids
        metadata["body_generation_steps"] = lexical_steps
        metadata["detected_language"] = detected_language
        if self._config.distribution_output_dir is not None:
            target = self._config.distribution_output_dir / f"{item.segment.id}-{self._export_index}.npz"
            self._export_index += 1
            metadata["distribution_file"] = str(export_full_logprobs(target, scores, ids, lexical_steps=lexical_steps, source_segment_id=item.segment.id, model_id=self.info.model_id))
        return TranscriptionSegment(item.segment.id, text, detected_language, None, None, scores.confidence, "raw_token_geometric_mean" if scores.confidence is not None else None, (), metadata)

    def transcribe(self, inputs: Sequence[TranscriptionInput], *, request: TranscriptionRequest) -> tuple[TranscriptionSegment, ...]:
        return tuple(self._one(item, request) for item in inputs)

    def close(self) -> None:
        # Qwen 运行时在部分 CUDA 环境中 allocator 清理会终止解释器；释放实例即可。
        self._wrapper = None
