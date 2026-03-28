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
    清理多余空白，得到适合拼接 prompt 的文本。

    参数:
        text: 原始文本。
    """
    return " ".join((text or "").strip().split())


class WhisperWord:
    """
    Whisper 词级时间戳对象。
    """

    def __init__(self, text: str, start: float | None, end: float | None, confidence: float | None):
        """
        初始化单个词对象。

        参数:
            text: 词文本。
            start: 词开始时间，单位秒；可能为空。
            end: 词结束时间，单位秒；可能为空。
            confidence: 词级置信度。当前实现使用片段平均置信度作为近似值。
        """
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence

    def to_dict(self) -> dict:
        """
        将词对象转换为字典。
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


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
        compression_ratio: float,
        padding_word_ratio: float,
        hallucination_risk: float,
        prompt_text: str,
        scores: dict,
        words: list[WhisperWord],
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
            language: Whisper 输出语言。
            avg_logprob: 片段平均对数概率。
            avg_confidence: 片段级置信度分数。
            confidence_source: 置信度来源，当前默认使用低显存启发式估计。
            compression_ratio: 文本压缩率，用于辅助识别重复或模板化输出。
            padding_word_ratio: 落在 padding 区间的词占比。
            hallucination_risk: 幻觉风险分数。
            prompt_text: 实际送入 Whisper 的文本提示。
            scores: 对应音频分片的声学分数。
            words: 词级时间戳对象列表。
            raw_result: 原始 Whisper 返回结果。
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
        self.compression_ratio = compression_ratio
        self.padding_word_ratio = padding_word_ratio
        self.hallucination_risk = hallucination_risk
        self.prompt_text = prompt_text
        self.scores = scores
        self.words = words
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
            "compression_ratio": self.compression_ratio,
            "padding_word_ratio": self.padding_word_ratio,
            "hallucination_risk": self.hallucination_risk,
            "prompt_text": self.prompt_text,
            "scores": self.scores,
            "words": [word.to_dict() for word in self.words],
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
            language: 目标语言。
            chunks: 分片转写结果列表。
        """
        self.music_path = music_path
        self.language = language
        self.chunks = chunks

    @property
    def merged_text(self) -> str:
        """
        获取整首歌的拼接转写文本。
        """
        return "\n".join(chunk.text for chunk in self.chunks if chunk.text)

    @property
    def stats(self) -> dict:
        """
        汇总整首歌转写统计。
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
        这个类只负责“对已经切好的片段做转写并收集证据”，不负责歌词对齐。
        上下文 prompt 只作为轻提示使用：
        1. 优先使用最近的高置信转写文本，保持上下文连续。
        2. 可选追加很短的歌词拼写提示。
        3. 对高风险片段自动减弱或关闭 prompt，避免把 Whisper 带成脑补模式。
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
        return_word_timestamps: bool = False,
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
            device: 推理设备，默认为自动选择 cuda / cpu。
            torch_dtype: 推理精度，默认为 cuda 使用 bfloat16，否则 float32。
            batch_size: 当前实现主要按片段逐个转写，此参数预留给后续批处理。
            max_new_tokens: 最大生成 token 数。
            return_word_timestamps: 是否请求词级时间戳。开启后显存与耗时都会明显上升。
            min_presence: 最低人声存在分数，低于该值可直接跳过。
            min_duration_sec: 最短转写片段时长，过短片段容易诱发幻觉。
            prompt_mode: prompt 策略，可选 `none` / `previous` / `hint` / `hybrid`。
            max_prompt_chars: prompt 总长度上限。
            max_previous_chunks: 最多回看多少个高置信历史片段。
            max_lyric_hint_chars: 歌词提示部分的长度上限。
        """
        self.model_id = model_id
        self.language = language
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.return_word_timestamps = return_word_timestamps
        self.min_presence = min_presence
        self.min_duration_sec = min_duration_sec
        self.prompt_mode = prompt_mode
        self.max_prompt_chars = max_prompt_chars
        self.max_previous_chunks = max_previous_chunks
        self.max_lyric_hint_chars = max_lyric_hint_chars

        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self):
        """
        懒加载 Whisper 模型、处理器和 pipeline。
        """
        if self.model is not None and self.processor is not None and self.pipe is not None:
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            dtype=self.torch_dtype,
            device=self.device,
        )

    def unload_model(self):
        """
        主动释放 Whisper 模型与 pipeline 占用的资源。
        """
        self.pipe = None
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
        将分片音频整理为 Whisper 所需的 16kHz 单声道输入。

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
            lyric_hint: 可为字符串、字符串列表，或 `LyricLineStamp` 这类歌词对象。
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
            lyric_hint: 可选的歌词拼写提示。
            off_center_risk: 当前片段的偏离中置风险，高风险时会自动减少 prompt。

        说明:
            这里只提供“最近高置信历史文本 + 短歌词提示”，
            不直接灌入整首歌词，避免模型把提示当答案进行脑补。
        """
        if self.prompt_mode == "none":
            return ""
        if off_center_risk >= 0.55:
            return ""

        prompt_parts: list[str] = []

        if self.prompt_mode in {"previous", "hybrid"}:
            usable_previous = []
            for chunk in reversed(previous_chunks):
                if not chunk.text:
                    continue
                if (chunk.avg_confidence or 0.0) < 0.45:
                    continue
                if chunk.hallucination_risk > 0.55:
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
                if not item:
                    continue
                if item in hint_items:
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
        generate_kwargs = {
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
                    prompt_ids = self.processor.get_prompt_ids(prompt_text)
                    generate_kwargs["prompt_ids"] = prompt_ids
                except Exception:
                    pass
            except Exception:
                pass
        return generate_kwargs

    def estimate_confidence(
        self,
        text: str,
        compression_ratio: float,
        padding_word_ratio: float,
        vocal_presence: float,
        off_center_risk: float,
        duration: float,
    ) -> float:
        """
        估计片段级置信度。

        参数:
            text: 转写文本。
            compression_ratio: 文本压缩率。
            padding_word_ratio: 落在 padding 区间的词占比。
            vocal_presence: 音频侧的人声存在分数。
            off_center_risk: 音频侧偏离中置风险。
            duration: 片段时长。

        说明:
            直接再跑一次 `generate(output_scores=True)` 会显著增加显存占用，
            对 6GB 显存设备不够友好。这里改为单次推理后的轻量估计分数，
            用于测试和后续降权，而不是精确 token logprob。
        """
        clean = _safe_text(text)
        if not clean:
            return 0.0

        density = len(clean) / max(duration, 1e-6)
        density_risk = _clamp01((density - 9.0) / 10.0)
        compression_risk = _clamp01((compression_ratio - 1.8) / 1.4)
        repeat_risk = self._repeat_ratio(clean)

        confidence = (
            0.52 * vocal_presence
            + 0.22 * (1.0 - off_center_risk)
            + 0.14 * (1.0 - padding_word_ratio)
            + 0.08 * (1.0 - compression_risk)
            + 0.04 * (1.0 - max(density_risk, repeat_risk))
        )
        return _clamp01(confidence)

    @staticmethod
    def _extract_words(prediction: dict, segment: dict, avg_confidence: float | None) -> tuple[list[WhisperWord], float]:
        """
        解析词级时间戳，并计算 padding 区词占比。

        参数:
            prediction: pipeline 返回结果。
            segment: 当前分片字典。
            avg_confidence: 片段平均置信度。
        """
        words: list[WhisperWord] = []
        outside_core = 0

        for item in prediction.get("chunks", []) or []:
            text = _safe_text(item.get("text", ""))
            timestamp = item.get("timestamp")
            if not text or not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                continue
            local_start, local_end = timestamp
            if local_start is None or local_end is None:
                continue

            abs_start = float(segment["start"] + local_start)
            abs_end = float(segment["start"] + local_end)
            if abs_end <= segment["core_start"] or abs_start >= segment["core_end"]:
                outside_core += 1
            words.append(
                WhisperWord(
                    text=text,
                    start=abs_start,
                    end=abs_end,
                    confidence=avg_confidence,
                )
            )

        padding_word_ratio = outside_core / max(len(words), 1) if words else 0.0
        return words, float(padding_word_ratio)

    def score_hallucination_risk(
        self,
        text: str,
        avg_confidence: float | None,
        compression_ratio: float,
        padding_word_ratio: float,
        off_center_risk: float,
        duration: float,
    ) -> float:
        """
        计算片段级幻觉风险。

        参数:
            text: 转写文本。
            avg_confidence: 平均置信度。
            compression_ratio: 文本压缩率。
            padding_word_ratio: 落在 padding 区间的词占比。
            off_center_risk: 音频侧的偏离中置风险。
            duration: 片段时长。

        说明:
            这里不是简单照搬 Whisper 的阈值，而是把声学风险和文本异常一起纳入。
            对高 `off_center_risk` 片段，即使转写出了文本，也会提高其幻觉风险。
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
            0.34 * inverse_conf
            + 0.24 * off_center_risk
            + 0.16 * padding_word_ratio
            + 0.14 * compression_risk
            + 0.07 * repeat_ratio
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
        pipe_kwargs = {
            "generate_kwargs": generate_kwargs,
        }
        if self.return_word_timestamps:
            pipe_kwargs["return_timestamps"] = "word"
        prediction = self.pipe(
            {"array": audio_16k, "sampling_rate": 16000},
            **pipe_kwargs,
        )

        text = _safe_text(prediction.get("text", ""))
        compression_ratio = self._compression_ratio(text)
        words, padding_word_ratio = self._extract_words(prediction, segment_data, None)
        avg_logprob = None
        avg_confidence = self.estimate_confidence(
            text=text,
            compression_ratio=compression_ratio,
            padding_word_ratio=padding_word_ratio,
            vocal_presence=float(scores.get("vocal_presence", 0.0)),
            off_center_risk=float(scores.get("off_center_risk", 0.0)),
            duration=float(segment_data["duration"]),
        )
        for word in words:
            word.confidence = avg_confidence
        hallucination_risk = self.score_hallucination_risk(
            text=text,
            avg_confidence=avg_confidence,
            compression_ratio=compression_ratio,
            padding_word_ratio=padding_word_ratio,
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
            confidence_source="heuristic",
            compression_ratio=compression_ratio,
            padding_word_ratio=padding_word_ratio,
            hallucination_risk=hallucination_risk,
            prompt_text=prompt_text,
            scores=dict(scores),
            words=words,
            raw_result=prediction,
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

        说明:
            这里只把 `vocal_presence` 足够高、且时长不太短的片段送入 Whisper。
            `off_center_risk` 不直接决定跳过，而是决定 prompt 强度和后续风险分数。
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
