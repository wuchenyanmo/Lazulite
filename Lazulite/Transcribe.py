import gzip
from difflib import SequenceMatcher
from collections.abc import Iterable

import librosa
import numpy as np
import torch

from Lazulite.Vocal import VocalAnalysisResult, VocalSegment


def _clamp01(value: float) -> float:
    """
    将浮点数裁剪到 [0, 1] 区间。

    参数:
        value: 输入分数。
    """
    return float(np.clip(value, 0.0, 1.0))


def _safe_text(text: str) -> str:
    """
    清理多余空白，得到适合展示和拼接的文本。

    参数:
        text: 原始文本。
    """
    return " ".join((text or "").strip().split())


class WhisperChunkResult:
    """
    单个分片的 Whisper 转写结果。
    """

    def __init__(
        self,
        segment_index: int,
        section_index: int | None,
        start: float,
        end: float,
        core_start: float,
        core_end: float,
        text: str,
        language: str | None,
        avg_logprob: float | None,
        avg_confidence: float | None,
        confidence_source: str,
        tokens: list[dict] | None,
        min_token_confidence: float | None,
        low_conf_token_ratio: float | None,
        compression_ratio: float,
        self_repeat_score: float,
        context_text: str,
        hallucination_risk: float,
        risk_components: dict,
        prompt_text: str,
        scores: dict,
        raw_result: dict | None = None,
    ):
        """
        初始化单个分片的转写结果。

        参数:
            segment_index: 分片序号。
            section_index: 所属长乐段序号。
            start: 分片开始时间，单位秒。
            end: 分片结束时间，单位秒。
            core_start: 核心区间开始时间，单位秒。
            core_end: 核心区间结束时间，单位秒。
            text: 转写文本。
            language: Whisper 语言设置；未指定时为 None。
            avg_logprob: 片段平均对数概率。
            avg_confidence: 片段平均置信度。
            confidence_source: 置信度来源。
            tokens: token 级详情列表，每个元素同时包含文本、置信度和时间戳。
            min_token_confidence: 最低 token 置信度。
            low_conf_token_ratio: 低置信 token 占比。
            compression_ratio: 文本压缩率。
            self_repeat_score: 句内重复分数，用于识别“一句话转写两次”。
            context_text: 仅供下一句 prompt 使用的裁剪后文本。
            hallucination_risk: 幻觉风险分数。
            risk_components: 幻觉风险的底层组件。
            prompt_text: 实际送入 Whisper 的文本提示。
            scores: 对应音频分片的声学分数。
            raw_result: 精简后的原始生成结果。
        """
        self.segment_index = segment_index
        self.section_index = section_index
        self.start = start
        self.end = end
        self.core_start = core_start
        self.core_end = core_end
        self.text = text
        self.language = language
        self.avg_logprob = avg_logprob
        self.avg_confidence = avg_confidence
        self.confidence_source = confidence_source
        self.tokens = tokens
        self.min_token_confidence = min_token_confidence
        self.low_conf_token_ratio = low_conf_token_ratio
        self.compression_ratio = compression_ratio
        self.self_repeat_score = self_repeat_score
        self.context_text = context_text
        self.hallucination_risk = hallucination_risk
        self.risk_components = risk_components
        self.prompt_text = prompt_text
        self.scores = scores
        self.raw_result = raw_result or {}

    @property
    def token_confidences(self) -> list[dict] | None:
        """
        兼容旧字段，返回 token 列表。
        """
        return self.tokens

    @property
    def token_timestamps(self) -> list[dict] | None:
        """
        兼容旧字段，返回 token 列表。
        """
        return self.tokens

    def to_dict(self) -> dict:
        """
        将分片转写结果转换为字典。
        """
        return {
            "segment_index": self.segment_index,
            "section_index": self.section_index,
            "start": self.start,
            "end": self.end,
            "core_start": self.core_start,
            "core_end": self.core_end,
            "text": self.text,
            "language": self.language,
            "avg_logprob": self.avg_logprob,
            "avg_confidence": self.avg_confidence,
            "confidence_source": self.confidence_source,
            "tokens": self.tokens,
            "min_token_confidence": self.min_token_confidence,
            "low_conf_token_ratio": self.low_conf_token_ratio,
            "compression_ratio": self.compression_ratio,
            "self_repeat_score": self.self_repeat_score,
            "context_text": self.context_text,
            "hallucination_risk": self.hallucination_risk,
            "risk_components": self.risk_components,
            "prompt_text": self.prompt_text,
            "scores": self.scores,
            "raw_result": self.raw_result,
        }


class WhisperTrackResult:
    """
    整首歌的 Whisper 转写聚合结果。
    """

    def __init__(self, music_path: str, language: str | None, chunks: list[WhisperChunkResult]):
        """
        初始化整首歌的转写结果。

        参数:
            music_path: 原始音频路径。
            language: 语言设置。
            chunks: 分片转写结果列表。
        """
        self.music_path = music_path
        self.language = language
        self.chunks = chunks

    @property
    def merged_text(self) -> str:
        """
        获取整首歌拼接后的文本。
        """
        return "\n".join(chunk.text for chunk in self.chunks if chunk.text)

    @property
    def stats(self) -> dict:
        """
        汇总整首歌的转写统计。
        """
        if not self.chunks:
            return {
                "chunk_count": 0,
                "non_empty_count": 0,
                "avg_confidence": None,
                "avg_hallucination_risk": None,
            }

        confidences = [chunk.avg_confidence for chunk in self.chunks if chunk.avg_confidence is not None]
        risks = [chunk.hallucination_risk for chunk in self.chunks]
        return {
            "chunk_count": len(self.chunks),
            "non_empty_count": sum(bool(chunk.text) for chunk in self.chunks),
            "avg_confidence": float(np.mean(confidences)) if confidences else None,
            "avg_hallucination_risk": float(np.mean(risks)) if risks else None,
        }

    def to_dict(self) -> dict:
        """
        将整首歌转写结果转换为字典。
        """
        return {
            "music_path": self.music_path,
            "language": self.language,
            "merged_text": self.merged_text,
            "stats": self.stats,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


class WhisperTranscriber:
    """
    基于 Whisper 的分片转写器。

    说明:
        当前实现只保留一条主路径：
        1. 用 `model.generate()` 直接生成文本。
        2. 同时读取 `output_scores=True` 得到 token 级置信度。
        3. 不再依赖词级时间戳，不再混用 pipeline 文本结果。
    """

    def __init__(
        self,
        model_id: str = "model/whisper-large-v3",
        language: str | None = None,
        task: str = "transcribe",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        batch_size: int = 1,
        max_new_tokens: int = 256,
        min_presence: float = 0.20,
        min_duration_sec: float = 0.80,
        prompt_mode: str = "hybrid",
        max_prompt_chars: int = 180,
        max_previous_chunks: int = 2,
        max_lyric_hint_chars: int = 80,
        num_candidates: int = 1,
        enable_token_timestamps: bool = True,
        disable_prompt_for_token_timestamps: bool = True,
        low_memory_mode: bool = False,
    ):
        """
        初始化 Whisper 转写器。

        参数:
            model_id: Whisper 模型路径或模型名。
            language: 指定语言；为 None 时让模型自动判断。
            task: Whisper 任务，通常为 `transcribe`。
            device: 推理设备。
            torch_dtype: 推理精度。
            batch_size: 预留参数，当前逐段转写。
            max_new_tokens: 最大生成 token 数。
            min_presence: 最低人声存在分数。
            min_duration_sec: 最短转写片段时长。
            prompt_mode: prompt 策略，可选 `none` / `previous` / `hint` / `hybrid`。
            max_prompt_chars: prompt 总长度上限。
            max_previous_chunks: 最多回看多少个历史片段作为 prompt。
            max_lyric_hint_chars: 歌词提示的长度上限。
            num_candidates: 同一分片重复转写的候选次数。
            enable_token_timestamps: 是否在同一次生成中请求 token 级时间戳。
            disable_prompt_for_token_timestamps: 是否在请求 token 时间戳时自动禁用 prompt；
                当前 transformers 的 Whisper 在 `prompt_ids + return_token_timestamps=True`
                组合下可能返回全 0 时间戳，因此默认关闭 prompt 以优先保证时间戳可用。
            low_memory_mode: 是否启用低显存模式；启用后在整条链路结束时建议主动卸载 Whisper。
        """
        self.model_id = model_id
        self.language = language
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.min_presence = min_presence
        self.min_duration_sec = min_duration_sec
        self.prompt_mode = prompt_mode
        self.max_prompt_chars = max_prompt_chars
        self.max_previous_chunks = max_previous_chunks
        self.max_lyric_hint_chars = max_lyric_hint_chars
        self.num_candidates = max(1, int(num_candidates))
        self.enable_token_timestamps = enable_token_timestamps
        self.disable_prompt_for_token_timestamps = disable_prompt_for_token_timestamps
        self.low_memory_mode = low_memory_mode

        self.model = None
        self.processor = None

    def load_model(self):
        """
        懒加载 Whisper 模型和处理器。
        """
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_kwargs = {
            "dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if self.device.startswith("cuda"):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def unload_model(self):
        """
        主动释放 Whisper 模型占用的资源。
        """
        self.processor = None
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model_if_needed(self):
        """
        按当前模式决定是否释放 Whisper 模型。
        """
        if self.low_memory_mode:
            self.unload_model()

    @staticmethod
    def _coerce_segment(segment: VocalSegment | dict) -> dict:
        """
        将分片对象统一转成字典。

        参数:
            segment: `VocalSegment` 或其字典形式。
        """
        if isinstance(segment, VocalSegment):
            return segment.to_dict()
        return segment

    @staticmethod
    def prepare_audio(audio: np.ndarray, input_sr: int, target_sr: int = 16000) -> np.ndarray:
        """
        将音频整理为 Whisper 所需的 16kHz 单声道输入。

        参数:
            audio: 输入音频。
            input_sr: 输入采样率。
            target_sr: 目标采样率。
        """
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        if input_sr != target_sr:
            y = librosa.resample(y, orig_sr=input_sr, target_sr=target_sr)
        return y.astype(np.float32)

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """
        计算文本压缩率，辅助识别重复模板或异常输出。

        参数:
            text: 转写文本。
        """
        clean = _safe_text(text)
        if len(clean) < 4:
            return 1.0
        raw = clean.encode("utf-8")
        compressed = gzip.compress(raw)
        return float(len(raw) / max(len(compressed), 1))

    @staticmethod
    def _repeat_ratio(text: str) -> float:
        """
        计算重复词比例。

        参数:
            text: 转写文本。
        """
        clean = _safe_text(text)
        if not clean:
            return 0.0
        tokens = clean.split()
        if len(tokens) <= 1:
            return 0.0
        repeated = sum(1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx - 1])
        return repeated / max(len(tokens) - 1, 1)

    @staticmethod
    def _extract_lyric_hint_items(lyric_hint) -> list[str]:
        """
        将歌词提示统一展开成短文本列表。

        参数:
            lyric_hint: 可为字符串、字符串列表，或歌词对象。
        """
        if lyric_hint is None:
            return []
        if isinstance(lyric_hint, str):
            return [_safe_text(lyric_hint)] if _safe_text(lyric_hint) else []
        if hasattr(lyric_hint, "get_alignment_texts"):
            values = lyric_hint.get_alignment_texts()
            return [_safe_text(item) for item in values if _safe_text(item)]
        if hasattr(lyric_hint, "normalized_lyrics"):
            values = getattr(lyric_hint, "normalized_lyrics")
            return [_safe_text(item) for item in values if _safe_text(item)]
        if hasattr(lyric_hint, "lyrics"):
            values = getattr(lyric_hint, "lyrics")
            return [_safe_text(item) for item in values if _safe_text(item)]
        if isinstance(lyric_hint, Iterable):
            return [_safe_text(str(item)) for item in lyric_hint if _safe_text(str(item))]
        return [_safe_text(str(lyric_hint))]

    def is_context_usable(self, chunk: WhisperChunkResult) -> bool:
        """
        判断历史分片是否适合作为上下文。

        参数:
            chunk: 已完成转写的历史分片。
        """
        if not chunk.text:
            return False
        if chunk.hallucination_risk >= 0.42:
            return False
        if chunk.min_token_confidence is not None and chunk.min_token_confidence < 0.10:
            return False
        if chunk.self_repeat_score > 0.88:
            return False
        return True

    @staticmethod
    def _trim_repeated_text_for_context(text: str) -> str:
        """
        将疑似“整句重复一次”的文本裁成更适合 prompt 的版本。

        参数:
            text: 原始转写文本。

        说明:
            这里不会修改最终转写结果，只在构造下一句上下文时使用。
            如果前后两半高度相似，则仅保留前半段，避免错误上下文连锁传播。
        """
        clean = _safe_text(text)
        if len(clean) < 8:
            return clean

        best_score = 0.0
        best_left = clean
        for split in range(max(3, len(clean) // 3), min(len(clean) - 3, (2 * len(clean)) // 3 + 1)):
            left = clean[:split].strip()
            right = clean[split:].strip()
            if len(left) < 3 or len(right) < 3:
                continue
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if length_ratio < 0.70:
                continue
            similarity = SequenceMatcher(None, left, right).ratio()
            candidate = similarity * length_ratio
            if candidate > best_score:
                best_score = candidate
                best_left = left

        if best_score >= 0.82:
            return best_left
        return clean

    def build_prompt_text(
        self,
        previous_chunks: list[WhisperChunkResult],
        lyric_hint=None,
        off_center_risk: float = 0.0,
    ) -> str:
        """
        构造 Whisper 的轻量文本 prompt。

        参数:
            previous_chunks: 已转写完成的历史片段。
            lyric_hint: 可选歌词提示。
            off_center_risk: 当前片段的偏离中置风险。
        """
        if self.prompt_mode == "none":
            return ""
        if off_center_risk >= 0.55:
            return ""

        prompt_parts: list[str] = []

        if self.prompt_mode in {"previous", "hybrid"}:
            usable_previous = []
            for chunk in reversed(previous_chunks):
                if not self.is_context_usable(chunk):
                    continue
                usable_previous.append(chunk.context_text or chunk.text)
                if len(usable_previous) >= self.max_previous_chunks:
                    break
            if usable_previous:
                prompt_parts.append(" ".join(reversed(usable_previous)))

        if self.prompt_mode in {"hint", "hybrid"} and off_center_risk < 0.40:
            hint_items = []
            used_chars = 0
            for item in self._extract_lyric_hint_items(lyric_hint):
                if not item or item in hint_items:
                    continue
                extra = len(item) + (1 if hint_items else 0)
                if used_chars + extra > self.max_lyric_hint_chars:
                    break
                hint_items.append(item)
                used_chars += extra
            if hint_items:
                prompt_parts.append(" / ".join(hint_items))

        prompt_text = _safe_text(" ".join(prompt_parts))
        if len(prompt_text) > self.max_prompt_chars:
            prompt_text = prompt_text[-self.max_prompt_chars:]
        return prompt_text

    def _build_generate_kwargs(self, prompt_text: str) -> dict:
        """
        构造 Whisper 生成参数。

        参数:
            prompt_text: 文本提示。
        """
        generate_kwargs: dict = {
            "task": self.task,
            "condition_on_prev_tokens": False,
        }
        if self.language:
            generate_kwargs["language"] = self.language
        can_use_prompt = not (self.enable_token_timestamps and self.disable_prompt_for_token_timestamps)
        if can_use_prompt and prompt_text and self.processor is not None and hasattr(self.processor, "get_prompt_ids"):
            try:
                prompt_ids = self.processor.get_prompt_ids(prompt_text, return_tensors="pt")
                if isinstance(prompt_ids, torch.Tensor):
                    prompt_ids = prompt_ids.to(self.device)
                generate_kwargs["prompt_ids"] = prompt_ids
            except TypeError:
                try:
                    generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(prompt_text)
                except Exception:
                    pass
            except Exception:
                pass
        return generate_kwargs

    def estimate_confidence(
        self,
        text: str,
        compression_ratio: float,
        vocal_presence: float,
        off_center_risk: float,
        duration: float,
    ) -> float:
        """
        在极少数无法拿到 generate 分数时估计置信度。

        参数:
            text: 转写文本。
            compression_ratio: 文本压缩率。
            vocal_presence: 音频侧的人声存在分数。
            off_center_risk: 音频侧偏离中置风险。
            duration: 片段时长。
        """
        clean = _safe_text(text)
        if not clean:
            return 0.0

        density = len(clean) / max(duration, 1e-6)
        density_risk = _clamp01((density - 9.0) / 10.0)
        compression_risk = _clamp01((compression_ratio - 1.8) / 1.4)
        repeat_risk = self._repeat_ratio(clean)

        confidence = (
            0.60 * vocal_presence
            + 0.24 * (1.0 - off_center_risk)
            + 0.10 * (1.0 - compression_risk)
            + 0.06 * (1.0 - max(density_risk, repeat_risk))
        )
        return _clamp01(confidence)

    def compute_generation_result(
        self,
        audio_16k: np.ndarray,
        generate_kwargs: dict,
    ) -> tuple[str | None, float | None, float | None, list[dict] | None, dict | None]:
        """
        使用 `model.generate()` 直接得到文本和 token 级置信度。

        参数:
            audio_16k: 16kHz 单声道音频。
            generate_kwargs: Whisper 生成参数。
        """
        if self.model is None or self.processor is None:
            return None, None, None, None, None

        try:
            with torch.inference_mode():
                inputs = self.processor(
                    audio_16k,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs.get("input_features")
                attention_mask = inputs.get("attention_mask")
                if input_features is None:
                    return None, None, None, None, None
                input_features = input_features.to(self.device, dtype=self.torch_dtype)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                generate_kwargs_with_timestamps = dict(generate_kwargs)
                generate_kwargs_with_timestamps["return_token_timestamps"] = self.enable_token_timestamps
                outputs = self.model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generate_kwargs_with_timestamps,
                )

            sequences = self._extract_generate_field(outputs, "sequences")
            scores = self._extract_generate_field(outputs, "scores")
            if sequences is None or not scores:
                return None, None, None, None, None

            decoded_text = _safe_text(
                self.processor.tokenizer.decode(
                    sequences[0],
                    skip_special_tokens=True,
                )
            )

            generated_ids = sequences[0, -len(scores):]
            token_logprobs = []
            tokens = []
            token_texts = []
            for step_index, step_scores in enumerate(scores):
                token_id = int(generated_ids[step_index])
                step_log_probs = torch.log_softmax(step_scores[0], dim=-1)
                token_logprob = float(step_log_probs[token_id].detach().cpu())
                token_confidence = float(np.clip(np.exp(token_logprob), 0.0, 1.0))
                token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
                token_texts.append(token_text)
                token_logprobs.append(token_logprob)
                tokens.append(
                    {
                        "token_id": token_id,
                        "token": token_text,
                        "text": token_text,
                        "logprob": token_logprob,
                        "confidence": token_confidence,
                        "start": None,
                        "end": None,
                    }
                )

            recovered_token_texts = self._recover_token_texts_from_sequence(generated_ids)
            usable_text_count = min(len(tokens), len(recovered_token_texts))
            for idx in range(usable_text_count):
                tokens[idx]["token"] = recovered_token_texts[idx]
                tokens[idx]["text"] = recovered_token_texts[idx]

            if self.enable_token_timestamps:
                timestamped_tokens = self._extract_token_timestamps(outputs, generated_ids, recovered_token_texts, tokens)
                if timestamped_tokens is not None:
                    tokens = timestamped_tokens
            avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else None
            avg_confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0)) if avg_logprob is not None else None
            return decoded_text, avg_logprob, avg_confidence, tokens, {"text": decoded_text}
        except Exception:
            return None, None, None, None, None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _extract_generate_field(outputs, key: str):
        """
        从 `generate()` 返回结果中提取字段，兼容对象和字典两种形式。

        参数:
            outputs: `generate()` 的返回值。
            key: 字段名。
        """
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    @staticmethod
    def _to_python_list(value):
        """
        将张量或数组安全转换成 Python 列表。

        参数:
            value: 待转换对象。
        """
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    def _extract_token_timestamps(
        self,
        outputs,
        generated_ids: torch.Tensor,
        token_texts: list[str],
        tokens: list[dict],
    ) -> list[dict] | None:
        """
        从 Whisper 生成结果中提取 token 级时间戳。

        参数:
            outputs: `generate()` 返回值。
            generated_ids: 当前片段生成出的 token id 序列。
            token_texts: 与生成 token 对应的解码文本。
            tokens: 与生成 token 对应的初始详情信息。

        说明:
            这里优先读取 `generate(return_token_timestamps=True)` 返回的时间戳；
            若当前 transformers 版本没有直接暴露该字段，则再尝试用 tokenizer offsets 兜底。
        """
        raw_timestamps = self._extract_generate_field(outputs, "token_timestamps")
        timestamp_values = self._to_python_list(raw_timestamps)
        if timestamp_values and isinstance(timestamp_values[0], list):
            timestamp_values = timestamp_values[0]

        if not timestamp_values:
            timestamp_values = self._decode_token_offsets(outputs)

        if not timestamp_values:
            return None

        token_ids = [int(token_id) for token_id in generated_ids.detach().cpu().tolist()]
        usable_count = min(len(token_ids), len(token_texts), len(tokens), len(timestamp_values))
        if usable_count <= 0:
            return None

        token_texts = self._recover_token_texts_from_sequence(generated_ids[:usable_count])
        token_entries = []
        prev_end = 0.0
        for idx in range(usable_count):
            token_text = token_texts[idx]
            token_info = dict(tokens[idx])
            timestamp_item = timestamp_values[idx]

            if isinstance(timestamp_item, dict):
                start = timestamp_item.get("start")
                end = timestamp_item.get("end")
            elif isinstance(timestamp_item, (list, tuple)) and len(timestamp_item) >= 2:
                start, end = timestamp_item[0], timestamp_item[1]
            else:
                start = prev_end
                end = timestamp_item

            if start is None:
                start = prev_end
            if end is None:
                end = start

            start = float(start)
            end = float(end)
            if end < start:
                end = start
            if start < prev_end:
                start = prev_end
            if end < start:
                end = start

            token_info["token_id"] = token_ids[idx]
            token_info["token"] = token_text
            token_info["text"] = token_text
            token_info["start"] = start
            token_info["end"] = end
            token_entries.append(token_info)
            prev_end = end

        token_entries = self._redistribute_leading_zero_timestamps(token_entries)
        return token_entries or None

    def _recover_token_texts_from_sequence(self, generated_ids: torch.Tensor) -> list[str]:
        """
        用前缀增量解码恢复每个 token 对应的文本片段。

        参数:
            generated_ids: 当前片段生成出的 token id 序列。

        说明:
            直接逐 token decode 在中日文场景下容易得到 `�`，因为单个 token
            可能只是一个多字节字符的一部分。这里改为“逐步解码整个前缀，再取增量”，
            以恢复更接近最终文本的 token 片段。
        """
        if self.processor is None:
            return []

        token_ids = [int(token_id) for token_id in generated_ids.detach().cpu().tolist()]
        recovered_texts: list[str] = []
        prev_text = ""
        for end_idx in range(1, len(token_ids) + 1):
            current_text = self.processor.tokenizer.decode(
                token_ids[:end_idx],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_text = current_text[len(prev_text):]
            recovered_texts.append(token_text)
            prev_text = current_text
        return recovered_texts

    @staticmethod
    def _redistribute_leading_zero_timestamps(token_entries: list[dict]) -> list[dict]:
        """
        将开头连续的零时长 token 均分到首个有效时间边界之前。

        参数:
            token_entries: token 时间戳条目列表。

        说明:
            Whisper 常把一句话前几个 token 的时间都压成 0。
            若后面已经出现首个有效边界，则把这段前导 token 均匀铺到该边界之前，
            以便后续做 chunk 内切分时能拿到更细的边界信息。
        """
        if not token_entries:
            return token_entries

        first_positive_idx = None
        first_positive_end = 0.0
        for idx, entry in enumerate(token_entries):
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
            if end > 0.0 or start > 0.0:
                first_positive_idx = idx
                first_positive_end = max(start, end)
                break

        if first_positive_idx is None or first_positive_idx == 0 or first_positive_end <= 0.0:
            return token_entries

        leading_count = first_positive_idx + 1
        step = first_positive_end / max(leading_count, 1)
        current = 0.0
        for idx in range(leading_count):
            next_time = first_positive_end if idx == leading_count - 1 else current + step
            token_entries[idx]["start"] = float(current)
            token_entries[idx]["end"] = float(max(next_time, current))
            current = next_time
        return token_entries

    def _decode_token_offsets(self, outputs) -> list | None:
        """
        使用 tokenizer offsets 作为 token 时间戳兜底。

        参数:
            outputs: `generate()` 返回值。
        """
        sequences = self._extract_generate_field(outputs, "sequences")
        if sequences is None or self.processor is None:
            return None
        try:
            decoded = self.processor.tokenizer.decode(
                sequences[0],
                skip_special_tokens=False,
                output_offsets=True,
                time_precision=0.02,
            )
        except Exception:
            return None

        if isinstance(decoded, dict):
            for key in ("offsets", "token_offsets", "chunks"):
                value = decoded.get(key)
                if value:
                    return value
        return None

    @staticmethod
    def token_confidence_stats(tokens: list[dict] | None) -> tuple[float | None, float | None]:
        """
        统计 token 级置信度的最低值与低分占比。

        参数:
            tokens: token 级详情列表。
        """
        if not tokens:
            return None, None
        values = [
            float(item["confidence"])
            for item in tokens
            if item.get("token", "").strip()
        ]
        if not values:
            return None, None
        min_conf = float(np.min(values))
        low_ratio = float(np.mean(np.array(values) < 0.35))
        return min_conf, low_ratio

    def score_hallucination_risk(
        self,
        text: str,
        avg_confidence: float | None,
        min_token_confidence: float | None,
        low_conf_token_ratio: float | None,
        compression_ratio: float,
        vocal_presence: float,
        off_center_risk: float,
        duration: float,
    ) -> tuple[float, float, dict]:
        """
        计算片段级幻觉风险。

        参数:
            text: 转写文本。
            avg_confidence: 平均置信度。
            min_token_confidence: 最低 token 置信度。
            low_conf_token_ratio: 低置信 token 占比。
            compression_ratio: 文本压缩率。
            vocal_presence: 音频侧人声存在分数。
            off_center_risk: 音频侧偏离中置风险。
            duration: 片段时长。
        """
        clean = _safe_text(text)
        if not clean:
            return 1.0, 0.0, {
                "avg_confidence_risk": 1.0,
                "min_token_risk": 1.0,
                "low_conf_ratio_risk": 1.0,
                "off_center_risk": 1.0,
                "presence_risk": 1.0,
                "self_repeat_risk": 0.0,
                "compression_risk": 0.0,
                "density_risk": 0.0,
            }

        avg_conf_risk = _clamp01(1.0 - (avg_confidence if avg_confidence is not None else 0.45))
        min_token_risk = self._min_token_penalty(min_token_confidence)
        low_conf_ratio_risk = self._low_conf_ratio_penalty(low_conf_token_ratio)
        self_repeat_score = self._self_repeat_score(clean)
        density_risk = self._density_penalty(clean, duration)
        compression_risk = _clamp01((compression_ratio - 1.8) / 1.4)
        off_center_penalty = self._off_center_penalty(off_center_risk)
        presence_penalty = self._presence_penalty(vocal_presence)

        risk_components = {
            "avg_confidence_risk": avg_conf_risk,
            "min_token_risk": min_token_risk,
            "low_conf_ratio_risk": low_conf_ratio_risk,
            "off_center_risk": off_center_penalty,
            "presence_risk": presence_penalty,
            "self_repeat_risk": self_repeat_score,
            "compression_risk": compression_risk,
            "density_risk": density_risk,
        }
        top_risks = sorted(risk_components.values(), reverse=True)
        risk = (
            0.50 * top_risks[0]
            + 0.30 * top_risks[1]
            + 0.20 * top_risks[2]
        )
        return _clamp01(risk), self_repeat_score, risk_components

    @staticmethod
    def _off_center_penalty(off_center_risk: float) -> float:
        """
        将保守的 `off_center_risk` 映射为更激进的风险惩罚。

        参数:
            off_center_risk: 音频侧偏离中置风险。
        """
        if off_center_risk <= 0.30:
            return _clamp01(0.25 * (off_center_risk / 0.30))
        normalized = (off_center_risk - 0.30) / 0.25
        return _clamp01(0.25 + 0.75 * (normalized ** 2))

    @staticmethod
    def _presence_penalty(vocal_presence: float) -> float:
        """
        将保守的 `vocal_presence` 映射为更激进的模糊人声惩罚。

        参数:
            vocal_presence: 音频侧人声存在分数。
        """
        if vocal_presence >= 0.70:
            return 0.0
        normalized = (0.70 - vocal_presence) / 0.18
        return _clamp01(normalized ** 2)

    @staticmethod
    def _min_token_penalty(min_token_confidence: float | None) -> float:
        """
        将最低 token 置信度映射为风险。

        参数:
            min_token_confidence: 最低 token 置信度。
        """
        if min_token_confidence is None:
            return 0.45
        if min_token_confidence >= 0.45:
            return 0.0
        normalized = (0.45 - min_token_confidence) / 0.35
        return _clamp01(normalized ** 1.6)

    @staticmethod
    def _low_conf_ratio_penalty(low_conf_token_ratio: float | None) -> float:
        """
        将低置信 token 占比映射为风险。

        参数:
            low_conf_token_ratio: 低置信 token 占比。
        """
        if low_conf_token_ratio is None:
            return 0.35
        if low_conf_token_ratio <= 0.10:
            return 0.0
        normalized = (low_conf_token_ratio - 0.10) / 0.50
        return _clamp01(normalized ** 1.4)

    @staticmethod
    def _density_penalty(text: str, duration: float) -> float:
        """
        将单位时长文本密度映射为风险。

        参数:
            text: 转写文本。
            duration: 片段时长。
        """
        density = len(text) / max(duration, 1e-6)
        return _clamp01((density - 9.0) / 10.0)

    @staticmethod
    def _self_repeat_score(text: str) -> float:
        """
        识别“同一句被 Whisper 连续转写两次”的情况。

        参数:
            text: 转写文本。

        说明:
            中文和日语不适合只按空格切词，这里直接在字符串层面检查：
            如果文本前半段与后半段高度相似，就提高重复分数。
        """
        clean = _safe_text(text)
        if len(clean) < 8:
            return 0.0

        best_score = 0.0
        for split in range(max(3, len(clean) // 3), min(len(clean) - 3, (2 * len(clean)) // 3 + 1)):
            left = clean[:split].strip()
            right = clean[split:].strip()
            if len(left) < 3 or len(right) < 3:
                continue
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if length_ratio < 0.70:
                continue
            similarity = SequenceMatcher(None, left, right).ratio()
            candidate = similarity * length_ratio
            if candidate > best_score:
                best_score = candidate
        return _clamp01(best_score)

    def transcribe_segment(
        self,
        segment: VocalSegment | dict,
        sr: int,
        segment_index: int,
        section_index: int | None = None,
        prompt_text: str = "",
    ) -> WhisperChunkResult:
        """
        对单个分片执行 Whisper 转写。

        参数:
            segment: 来自 `VocalAnalyzer` 的分片对象。
            sr: 原始采样率。
            segment_index: 分片序号。
            section_index: 所属长乐段序号。
            prompt_text: 文本提示。
        """
        self.load_model()
        segment_data = self._coerce_segment(segment)
        scores = segment_data["scores"]
        audio_16k = self.prepare_audio(segment_data["audio"], input_sr=sr)
        effective_prompt_text = prompt_text
        prompt_disabled_for_timestamps = False
        if self.enable_token_timestamps and self.disable_prompt_for_token_timestamps and prompt_text:
            effective_prompt_text = ""
            prompt_disabled_for_timestamps = True

        generate_kwargs = self._build_generate_kwargs(effective_prompt_text)
        candidates: list[WhisperChunkResult] = []
        for candidate_index in range(self.num_candidates):
            text, avg_logprob, avg_confidence, tokens, raw_result = self.compute_generation_result(
                audio_16k=audio_16k,
                generate_kwargs=generate_kwargs,
            )
            text = text or ""
            min_token_confidence, low_conf_token_ratio = self.token_confidence_stats(tokens)
            compression_ratio = self._compression_ratio(text)

            confidence_source = "generate_scores"
            if avg_confidence is None:
                confidence_source = "heuristic"
                avg_confidence = self.estimate_confidence(
                    text=text,
                    compression_ratio=compression_ratio,
                    vocal_presence=float(scores.get("vocal_presence", 0.0)),
                    off_center_risk=float(scores.get("off_center_risk", 0.0)),
                    duration=float(segment_data["duration"]),
                )

            hallucination_risk, self_repeat_score, risk_components = self.score_hallucination_risk(
                text=text,
                avg_confidence=avg_confidence,
                min_token_confidence=min_token_confidence,
                low_conf_token_ratio=low_conf_token_ratio,
                compression_ratio=compression_ratio,
                vocal_presence=float(scores.get("vocal_presence", 0.0)),
                off_center_risk=float(scores.get("off_center_risk", 0.0)),
                duration=float(segment_data["duration"]),
            )
            context_text = self._trim_repeated_text_for_context(text)
            raw_result = dict(raw_result or {})
            raw_result["candidate_index"] = candidate_index
            if prompt_disabled_for_timestamps:
                raw_result["requested_prompt_text"] = prompt_text
                raw_result["prompt_disabled_for_token_timestamps"] = True

            candidates.append(
                WhisperChunkResult(
                    segment_index=segment_index,
                    section_index=section_index,
                    start=float(segment_data["start"]),
                    end=float(segment_data["end"]),
                    core_start=float(segment_data["core_start"]),
                    core_end=float(segment_data["core_end"]),
                    text=text,
                    language=self.language,
                    avg_logprob=avg_logprob,
                    avg_confidence=avg_confidence,
                    confidence_source=confidence_source,
                    tokens=tokens,
                    min_token_confidence=min_token_confidence,
                    low_conf_token_ratio=low_conf_token_ratio,
                    compression_ratio=compression_ratio,
                    self_repeat_score=self_repeat_score,
                    context_text=context_text,
                    hallucination_risk=hallucination_risk,
                    risk_components=risk_components,
                    prompt_text=effective_prompt_text,
                    scores=dict(scores),
                    raw_result=raw_result,
                )
            )

        best_index = self._select_best_candidate_index(candidates)
        best = candidates[best_index]
        best.raw_result = dict(best.raw_result or {})
        best.raw_result["num_candidates"] = len(candidates)
        best.raw_result["selected_index"] = best_index
        if len(candidates) > 1:
            best.raw_result["candidate_texts"] = [candidate.text for candidate in candidates]
        return best

    @staticmethod
    def _candidate_quality(candidate: WhisperChunkResult) -> float:
        """
        计算候选转写结果的综合质量分数，分数越高越优。

        参数:
            candidate: 单个候选转写结果。
        """
        avg_confidence = float(candidate.avg_confidence or 0.0)
        avg_logprob = float(candidate.avg_logprob or -10.0)
        min_token = float(candidate.min_token_confidence or 0.0)
        low_conf_ratio = float(candidate.low_conf_token_ratio or 0.0)
        score = (
            1.15 * avg_confidence
            + 0.10 * avg_logprob
            + 0.22 * min_token
            - 1.25 * float(candidate.hallucination_risk)
            - 0.24 * float(candidate.self_repeat_score)
            - 0.16 * low_conf_ratio
        )
        if not candidate.text:
            score -= 1.0
        return score

    def _select_best_candidate_index(self, candidates: list[WhisperChunkResult]) -> int:
        """
        在多次转写结果中选出质量最好的候选。

        参数:
            candidates: 同一分片的多个候选结果。
        """
        if not candidates:
            return 0
        return max(
            range(len(candidates)),
            key=lambda idx: (
                self._candidate_quality(candidates[idx]),
                float(candidates[idx].avg_confidence or 0.0),
                -float(candidates[idx].hallucination_risk),
            ),
        )

    def transcribe_analysis(
        self,
        analysis_result: VocalAnalysisResult | dict,
        lyric_hint=None,
        segment_indices: set[int] | None = None,
    ) -> WhisperTrackResult:
        """
        对 `VocalAnalyzer` 的分析结果做批量分片转写。

        参数:
            analysis_result: `VocalAnalyzer.analyze_file()` 的结果对象或字典。
            lyric_hint: 可选的轻量歌词提示。
            segment_indices: 若指定，则仅转写这些 1-based segment 序号。
        """
        if isinstance(analysis_result, VocalAnalysisResult):
            data = analysis_result.to_dict()
        else:
            data = analysis_result

        previous_chunks: list[WhisperChunkResult] = []
        chunk_results: list[WhisperChunkResult] = []
        segments = data.get("segments", [])
        sections = data.get("sections", [])
        sr = int(data["sr"])
        section_index_by_segment: dict[int, int] = {}
        for section in sections:
            for item in section.get("segment_indices", []):
                section_index_by_segment[int(item)] = int(section.get("section_index", 0) or 0)

        for index, segment in enumerate(segments, start=1):
            if segment_indices is not None and index not in segment_indices:
                continue
            scores = segment.get("scores", {})
            if float(scores.get("vocal_presence", 0.0)) < self.min_presence:
                continue
            if float(segment.get("duration", 0.0)) < self.min_duration_sec:
                continue

            prompt_text = self.build_prompt_text(
                previous_chunks=previous_chunks,
                lyric_hint=lyric_hint,
                off_center_risk=float(scores.get("off_center_risk", 0.0)),
            )
            chunk = self.transcribe_segment(
                segment=segment,
                sr=sr,
                segment_index=index,
                section_index=section_index_by_segment.get(index) or None,
                prompt_text=prompt_text,
            )
            chunk_results.append(chunk)
            previous_chunks.append(chunk)

        return WhisperTrackResult(
            music_path=data.get("music_path", ""),
            language=self.language,
            chunks=chunk_results,
        )
