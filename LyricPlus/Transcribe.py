import gzip
from collections.abc import Iterable

import librosa
import numpy as np
import torch

from LyricPlus.Vocal import VocalAnalysisResult, VocalSegment


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
        start: float,
        end: float,
        core_start: float,
        core_end: float,
        text: str,
        language: str | None,
        avg_logprob: float | None,
        avg_confidence: float | None,
        confidence_source: str,
        token_confidences: list[dict] | None,
        min_token_confidence: float | None,
        low_conf_token_ratio: float | None,
        compression_ratio: float,
        hallucination_risk: float,
        prompt_text: str,
        scores: dict,
        raw_result: dict | None = None,
    ):
        """
        初始化单个分片的转写结果。

        参数:
            segment_index: 分片序号。
            start: 分片开始时间，单位秒。
            end: 分片结束时间，单位秒。
            core_start: 核心区间开始时间，单位秒。
            core_end: 核心区间结束时间，单位秒。
            text: 转写文本。
            language: Whisper 语言设置；未指定时为 None。
            avg_logprob: 片段平均对数概率。
            avg_confidence: 片段平均置信度。
            confidence_source: 置信度来源。
            token_confidences: token 级置信度列表。
            min_token_confidence: 最低 token 置信度。
            low_conf_token_ratio: 低置信 token 占比。
            compression_ratio: 文本压缩率。
            hallucination_risk: 幻觉风险分数。
            prompt_text: 实际送入 Whisper 的文本提示。
            scores: 对应音频分片的声学分数。
            raw_result: 精简后的原始生成结果。
        """
        self.segment_index = segment_index
        self.start = start
        self.end = end
        self.core_start = core_start
        self.core_end = core_end
        self.text = text
        self.language = language
        self.avg_logprob = avg_logprob
        self.avg_confidence = avg_confidence
        self.confidence_source = confidence_source
        self.token_confidences = token_confidences
        self.min_token_confidence = min_token_confidence
        self.low_conf_token_ratio = low_conf_token_ratio
        self.compression_ratio = compression_ratio
        self.hallucination_risk = hallucination_risk
        self.prompt_text = prompt_text
        self.scores = scores
        self.raw_result = raw_result or {}

    def to_dict(self) -> dict:
        """
        将分片转写结果转换为字典。
        """
        return {
            "segment_index": self.segment_index,
            "start": self.start,
            "end": self.end,
            "core_start": self.core_start,
            "core_end": self.core_end,
            "text": self.text,
            "language": self.language,
            "avg_logprob": self.avg_logprob,
            "avg_confidence": self.avg_confidence,
            "confidence_source": self.confidence_source,
            "token_confidences": self.token_confidences,
            "min_token_confidence": self.min_token_confidence,
            "low_conf_token_ratio": self.low_conf_token_ratio,
            "compression_ratio": self.compression_ratio,
            "hallucination_risk": self.hallucination_risk,
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
        if (chunk.avg_confidence or 0.0) < 0.50:
            return False
        if chunk.hallucination_risk > 0.45:
            return False
        if float(chunk.scores.get("off_center_risk", 0.0)) > 0.35:
            return False
        if chunk.min_token_confidence is not None and chunk.min_token_confidence < 0.18:
            return False
        if chunk.low_conf_token_ratio is not None and chunk.low_conf_token_ratio > 0.35:
            return False
        return True

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
                usable_previous.append(chunk.text)
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
        if prompt_text and self.processor is not None and hasattr(self.processor, "get_prompt_ids"):
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
                inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.get("input_features")
                if input_features is None:
                    return None, None, None, None, None
                input_features = input_features.to(self.device, dtype=self.torch_dtype)

                outputs = self.model.generate(
                    input_features=input_features,
                    max_new_tokens=self.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generate_kwargs,
                )

            sequences = getattr(outputs, "sequences", None)
            scores = getattr(outputs, "scores", None)
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
            token_confidences = []
            for step_index, step_scores in enumerate(scores):
                token_id = int(generated_ids[step_index])
                step_log_probs = torch.log_softmax(step_scores[0], dim=-1)
                token_logprob = float(step_log_probs[token_id].detach().cpu())
                token_confidence = float(np.clip(np.exp(token_logprob), 0.0, 1.0))
                token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
                token_logprobs.append(token_logprob)
                token_confidences.append(
                    {
                        "token_id": token_id,
                        "token": token_text,
                        "logprob": token_logprob,
                        "confidence": token_confidence,
                    }
                )

            avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else None
            avg_confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0)) if avg_logprob is not None else None
            return decoded_text, avg_logprob, avg_confidence, token_confidences, {"text": decoded_text}
        except Exception:
            return None, None, None, None, None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def token_confidence_stats(token_confidences: list[dict] | None) -> tuple[float | None, float | None]:
        """
        统计 token 级置信度的最低值与低分占比。

        参数:
            token_confidences: token 级置信度列表。
        """
        if not token_confidences:
            return None, None
        values = [
            float(item["confidence"])
            for item in token_confidences
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
        compression_ratio: float,
        off_center_risk: float,
        duration: float,
    ) -> float:
        """
        计算片段级幻觉风险。

        参数:
            text: 转写文本。
            avg_confidence: 平均置信度。
            compression_ratio: 文本压缩率。
            off_center_risk: 音频侧偏离中置风险。
            duration: 片段时长。
        """
        clean = _safe_text(text)
        if not clean:
            return 1.0

        inverse_conf = 1.0 - (avg_confidence if avg_confidence is not None else 0.45)
        repeat_ratio = self._repeat_ratio(clean)
        density = len(clean) / max(duration, 1e-6)
        density_risk = _clamp01((density - 9.0) / 10.0)
        compression_risk = _clamp01((compression_ratio - 1.8) / 1.4)

        risk = (
            0.42 * inverse_conf
            + 0.28 * off_center_risk
            + 0.17 * compression_risk
            + 0.08 * repeat_ratio
            + 0.05 * density_risk
        )
        return _clamp01(risk)

    def transcribe_segment(
        self,
        segment: VocalSegment | dict,
        sr: int,
        segment_index: int,
        prompt_text: str = "",
    ) -> WhisperChunkResult:
        """
        对单个分片执行 Whisper 转写。

        参数:
            segment: 来自 `VocalAnalyzer` 的分片对象。
            sr: 原始采样率。
            segment_index: 分片序号。
            prompt_text: 文本提示。
        """
        self.load_model()
        segment_data = self._coerce_segment(segment)
        scores = segment_data["scores"]
        audio_16k = self.prepare_audio(segment_data["audio"], input_sr=sr)
        generate_kwargs = self._build_generate_kwargs(prompt_text)

        text, avg_logprob, avg_confidence, token_confidences, raw_result = self.compute_generation_result(
            audio_16k=audio_16k,
            generate_kwargs=generate_kwargs,
        )
        text = text or ""
        min_token_confidence, low_conf_token_ratio = self.token_confidence_stats(token_confidences)
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

        hallucination_risk = self.score_hallucination_risk(
            text=text,
            avg_confidence=avg_confidence,
            compression_ratio=compression_ratio,
            off_center_risk=float(scores.get("off_center_risk", 0.0)),
            duration=float(segment_data["duration"]),
        )

        return WhisperChunkResult(
            segment_index=segment_index,
            start=float(segment_data["start"]),
            end=float(segment_data["end"]),
            core_start=float(segment_data["core_start"]),
            core_end=float(segment_data["core_end"]),
            text=text,
            language=self.language,
            avg_logprob=avg_logprob,
            avg_confidence=avg_confidence,
            confidence_source=confidence_source,
            token_confidences=token_confidences,
            min_token_confidence=min_token_confidence,
            low_conf_token_ratio=low_conf_token_ratio,
            compression_ratio=compression_ratio,
            hallucination_risk=hallucination_risk,
            prompt_text=prompt_text,
            scores=dict(scores),
            raw_result=raw_result,
        )

    def transcribe_analysis(
        self,
        analysis_result: VocalAnalysisResult | dict,
        lyric_hint=None,
    ) -> WhisperTrackResult:
        """
        对 `VocalAnalyzer` 的分析结果做批量分片转写。

        参数:
            analysis_result: `VocalAnalyzer.analyze_file()` 的结果对象或字典。
            lyric_hint: 可选的轻量歌词提示。
        """
        if isinstance(analysis_result, VocalAnalysisResult):
            data = analysis_result.to_dict()
        else:
            data = analysis_result

        previous_chunks: list[WhisperChunkResult] = []
        chunk_results: list[WhisperChunkResult] = []
        segments = data.get("segments", [])
        sr = int(data["sr"])

        for index, segment in enumerate(segments, start=1):
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
                prompt_text=prompt_text,
            )
            chunk_results.append(chunk)
            previous_chunks.append(chunk)

        return WhisperTrackResult(
            music_path=data.get("music_path", ""),
            language=self.language,
            chunks=chunk_results,
        )
