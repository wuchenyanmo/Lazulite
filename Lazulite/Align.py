from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from Lazulite.Lyric import LyricLineStamp, LyricTokenLine
from Lazulite.Transcribe import WhisperChunkResult, WhisperTrackResult


class AlignedLyricLine:
    """
    单行歌词的约束对齐结果。
    """

    def __init__(
        self,
        line_index: int,
        line: LyricTokenLine,
        normalized_text: str,
        chunk_indices: list[int],
        chunk_text: str | None,
        start: float | None,
        end: float | None,
        similarity: float,
        score: float,
        confidence: float | None,
        hallucination_risk: float | None,
    ):
        """
        初始化单行对齐结果。

        参数:
            line_index: 歌词行序号。
            line: 原始歌词行对象。
            normalized_text: 歌词规范化文本。
            chunk_indices: 匹配到的转写 chunk 序号列表。
            chunk_text: 匹配到的转写文本，可由多个 chunk 拼接而成。
            start: 对齐后的开始时间。
            end: 对齐后的结束时间。
            similarity: 文本相似度。
            score: 动态规划中的局部匹配分数。
            confidence: 转写 chunk 的平均置信度。
            hallucination_risk: 转写 chunk 的平均幻觉风险。
        """
        self.line_index = line_index
        self.line = line
        self.normalized_text = normalized_text
        self.chunk_indices = chunk_indices
        self.chunk_text = chunk_text
        self.start = start
        self.end = end
        self.similarity = similarity
        self.score = score
        self.confidence = confidence
        self.hallucination_risk = hallucination_risk

    @property
    def chunk_index(self) -> int | None:
        """
        兼容旧字段，返回首个匹配 chunk 序号。
        """
        return self.chunk_indices[0] if self.chunk_indices else None

    def to_dict(self) -> dict:
        """
        将对齐结果转换为字典。
        """
        return {
            "line_index": self.line_index,
            "timestamp": self.line.timestamp,
            "text": self.line.text,
            "normalized_text": self.normalized_text,
            "singer": self.line.singer,
            "translation": self.line.translation,
            "chunk_index": self.chunk_index,
            "chunk_indices": self.chunk_indices,
            "chunk_text": self.chunk_text,
            "start": self.start,
            "end": self.end,
            "similarity": self.similarity,
            "score": self.score,
            "confidence": self.confidence,
            "hallucination_risk": self.hallucination_risk,
        }


class LyricAlignmentResult:
    """
    整首歌的歌词约束对齐结果。
    """

    def __init__(
        self,
        music_path: str,
        items: list[AlignedLyricLine],
        strategy: str = "dp",
        details: dict | None = None,
    ):
        """
        初始化整首歌对齐结果。

        参数:
            music_path: 原始音频路径。
            items: 行级对齐结果列表。
            strategy: 当前结果使用的对齐策略。
            details: 额外调试信息。
        """
        self.music_path = music_path
        self.items = items
        self.strategy = strategy
        self.details = details or {}

    @property
    def matched_count(self) -> int:
        """
        获取成功匹配到转写 chunk 的歌词行数。
        """
        return sum(bool(item.chunk_indices) for item in self.items)

    @property
    def stats(self) -> dict:
        """
        汇总对齐统计。
        """
        similarities = [item.similarity for item in self.items if item.chunk_indices]
        scores = [item.score for item in self.items if item.chunk_indices]
        confidences = [item.confidence for item in self.items if item.confidence is not None]
        return {
            "line_count": len(self.items),
            "matched_count": self.matched_count,
            "match_ratio": self.matched_count / max(len(self.items), 1),
            "avg_similarity": float(np.mean(similarities)) if similarities else None,
            "avg_score": float(np.mean(scores)) if scores else None,
            "avg_confidence": float(np.mean(confidences)) if confidences else None,
            "strategy": self.strategy,
        }

    def to_dict(self) -> dict:
        """
        将整首歌对齐结果转换为字典。
        """
        return {
            "music_path": self.music_path,
            "strategy": self.strategy,
            "stats": self.stats,
            "details": self.details,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class _HybridOffsetAnchor:
    line_index: int
    merged_section_index: int | None
    delta: float
    score: float
    chunk_confidence: float | None


class LyricAligner:
    """
    歌词与转写结果的约束对齐器。

    说明:
        这里使用单调动态规划对齐：
        1. 歌词行顺序不能乱。
        2. 转写 chunk 顺序不能乱。
        3. 允许跳过歌词行或跳过 chunk。
        4. 允许一个歌词行跨多个 chunk，也允许一个 chunk 覆盖多行歌词。
        5. 匹配分数综合文本相似度、转写置信度和风险惩罚。
    """

    def __init__(
        self,
        skip_line_penalty: float = 0.34,
        skip_chunk_penalty: float = 0.24,
        min_match_similarity: float = 0.08,
        max_chunks_per_line: int = 2,
        max_lines_per_chunk: int = 4,
    ):
        """
        初始化对齐器。

        参数:
            skip_line_penalty: 跳过歌词行的代价。
            skip_chunk_penalty: 跳过转写 chunk 的代价。
            min_match_similarity: 允许视为“弱匹配”的最低相似度。
            max_chunks_per_line: 单行歌词最多允许合并多少个 chunk。
            max_lines_per_chunk: 单个 chunk 最多允许覆盖多少行歌词。
        """
        self.skip_line_penalty = skip_line_penalty
        self.skip_chunk_penalty = skip_chunk_penalty
        self.min_match_similarity = min_match_similarity
        self.max_chunks_per_line = max_chunks_per_line
        self.max_lines_per_chunk = max_lines_per_chunk
        self.token_skip_penalty = 0.018
        self.boundary_trim_penalty = 0.010

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        """
        构造字符 n-gram 集合。

        参数:
            text: 输入文本。
            n: n-gram 长度。
        """
        if len(text) < n:
            return {text} if text else set()
        return {text[idx: idx + n] for idx in range(len(text) - n + 1)}

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        """
        对可能含空值的列表求均值。

        参数:
            values: 浮点数列表。
        """
        clean = [float(value) for value in values if value is not None]
        return float(np.mean(clean)) if clean else None

    @staticmethod
    def _coerce_chunks(transcription) -> tuple[str, list[WhisperChunkResult]]:
        """
        将转写结果统一成 chunk 列表。

        参数:
            transcription: `WhisperTrackResult` 或其字典形式。
        """
        if isinstance(transcription, WhisperTrackResult):
            return transcription.music_path, transcription.chunks

        music_path = transcription.get("music_path", "")
        chunk_list: list[WhisperChunkResult] = []
        for item in transcription.get("chunks", []):
            if isinstance(item, WhisperChunkResult):
                chunk_list.append(item)
                continue
            chunk_list.append(
                WhisperChunkResult(
                    segment_index=item["segment_index"],
                    section_index=item.get("section_index"),
                    start=item["start"],
                    end=item["end"],
                    core_start=item["core_start"],
                    core_end=item["core_end"],
                    text=item.get("text", ""),
                    language=item.get("language"),
                    avg_logprob=item.get("avg_logprob"),
                    avg_confidence=item.get("avg_confidence"),
                    confidence_source=item.get("confidence_source", ""),
                    tokens=item.get("tokens") or LyricAligner._merge_legacy_tokens(item),
                    min_token_confidence=item.get("min_token_confidence"),
                    low_conf_token_ratio=item.get("low_conf_token_ratio"),
                    compression_ratio=item.get("compression_ratio", 1.0),
                    self_repeat_score=item.get("self_repeat_score", 0.0),
                    context_text=item.get("context_text", item.get("text", "")),
                    hallucination_risk=item.get("hallucination_risk", 1.0),
                    risk_components=item.get("risk_components", {}),
                    prompt_text=item.get("prompt_text", ""),
                    scores=item.get("scores", {}),
                    raw_result=item.get("raw_result"),
                )
            )
        return music_path, chunk_list

    @staticmethod
    def _merge_legacy_tokens(item: dict) -> list[dict] | None:
        """
        将旧版分开的 token 字段合并成统一结构。

        参数:
            item: chunk 字典。
        """
        confidences = item.get("token_confidences") or []
        timestamps = item.get("token_timestamps") or []
        if not confidences and not timestamps:
            return None

        merged = []
        usable_count = max(len(confidences), len(timestamps))
        for idx in range(usable_count):
            info = {}
            if idx < len(confidences):
                info.update(confidences[idx] or {})
            if idx < len(timestamps):
                info.update(timestamps[idx] or {})
            merged.append(info)
        return merged

    @staticmethod
    def _merge_penalty(line_count: int, chunk_count: int) -> float:
        """
        计算窗口合并惩罚。

        参数:
            line_count: 歌词行数。
            chunk_count: 转写 chunk 数。
        """
        penalty = 0.0
        if line_count > 1:
            penalty += 0.05 * (line_count - 1)
        if chunk_count > 1:
            penalty += 0.04 * (chunk_count - 1)
        return penalty

    @staticmethod
    def _join_token_texts(entries: list[dict]) -> str:
        """
        将 token 条目拼成可用于匹配的文本。

        参数:
            entries: token 条目列表。
        """
        text = "".join(str(entry.get("text", entry.get("token", ""))) for entry in entries)
        return " ".join(text.strip().split())

    @staticmethod
    def _ordered_unique(values: list[int]) -> list[int]:
        """
        按原顺序去重。

        参数:
            values: 整数列表。
        """
        seen = set()
        result = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _weighted_median(values: list[tuple[float, float]]) -> float | None:
        clean = [
            (float(value), max(float(weight), 0.0))
            for value, weight in values
            if value is not None and weight is not None and float(weight) > 0.0
        ]
        if not clean:
            return None
        clean.sort(key=lambda item: item[0])
        total_weight = sum(weight for _, weight in clean)
        threshold = total_weight / 2.0
        cumulative = 0.0
        for value, weight in clean:
            cumulative += weight
            if cumulative >= threshold:
                return float(value)
        return float(clean[-1][0])

    @staticmethod
    def _prefix_similarity(a: str, b: str, prefix_chars: int = 12) -> float:
        a = LyricLineStamp.normalize_lyric_text(a)
        b = LyricLineStamp.normalize_lyric_text(b)
        if not a or not b:
            return 0.0
        usable = min(len(a), len(b), max(prefix_chars, 1))
        return float(SequenceMatcher(None, a[:usable], b[:usable]).ratio())

    def _item_alignment_reliability(self, item: AlignedLyricLine) -> float:
        prefix_similarity = self._prefix_similarity(
            item.line.normalized_text,
            item.chunk_text or "",
        )
        confidence = self._clip(float(item.confidence or 0.0), 0.0, 1.0)
        hallucination = self._clip(float(item.hallucination_risk or 1.0), 0.0, 1.0)
        reliability = (
            0.58 * prefix_similarity
            + 0.27 * confidence
            + 0.15 * (1.0 - hallucination)
        )
        return self._clip(reliability, 0.0, 1.0)

    @staticmethod
    def _infer_item_section_index(
        item: AlignedLyricLine,
        chunk_section_by_segment: dict[int, int | None],
    ) -> int | None:
        section_indices = [
            chunk_section_by_segment.get(segment_index)
            for segment_index in item.chunk_indices
            if chunk_section_by_segment.get(segment_index) is not None
        ]
        if not section_indices:
            return None
        counts: dict[int, int] = {}
        for section_index in section_indices:
            counts[int(section_index)] = counts.get(int(section_index), 0) + 1
        return max(counts.items(), key=lambda pair: pair[1])[0]

    @staticmethod
    def _smooth_monotonic_timestamps(
        starts: list[float | None],
        min_gap: float = 0.02,
    ) -> list[float | None]:
        result = [None if value is None else float(value) for value in starts]
        previous = None
        for idx, value in enumerate(result):
            if value is None:
                continue
            if previous is not None and value < previous + min_gap:
                value = previous + min_gap
                result[idx] = value
            previous = value
        return result

    def _flatten_chunk_token_entries(self, chunks: list[WhisperChunkResult]) -> list[dict]:
        """
        将若干 chunk 的 token 时间戳展开成带绝对时间的统一序列。

        参数:
            chunks: 连续转写 chunk 列表。
        """
        entries: list[dict] = []
        for chunk in chunks:
            token_entries = chunk.token_timestamps or []
            if token_entries:
                for item in token_entries:
                    text = str(item.get("text", item.get("token", "")))
                    if text.startswith("<|") and text.endswith("|>"):
                        continue
                    if not text.strip():
                        continue
                    start = item.get("start")
                    end = item.get("end")
                    if start is None and end is None:
                        continue
                    start = float(chunk.start if start is None else chunk.start + float(start))
                    end = float(start if end is None else chunk.start + float(end))
                    if end < start:
                        end = start
                    entries.append(
                        {
                            "segment_index": chunk.segment_index,
                            "token_id": item.get("token_id"),
                            "token": item.get("token", text),
                            "text": text,
                            "start": start,
                            "end": end,
                        }
                    )
                continue

            if chunk.text:
                entries.append(
                    {
                        "segment_index": chunk.segment_index,
                        "token_id": None,
                        "token": chunk.text,
                        "text": chunk.text,
                        "start": float(chunk.start),
                        "end": float(chunk.end),
                    }
                )
        return entries

    def _best_token_boundary_split(
        self,
        lyric_lines: list[LyricTokenLine],
        chunks: list[WhisperChunkResult],
        confidence: float,
        hallucination: float,
        presence: float,
        off_center: float,
    ) -> dict | None:
        """
        在 token 边界上为多行歌词寻找最佳切分。

        参数:
            lyric_lines: 连续歌词行列表。
            chunks: 连续转写 chunk 列表。
            confidence: 窗口平均置信度。
            hallucination: 窗口平均幻觉风险。
            presence: 窗口平均人声存在分数。
            off_center: 窗口平均偏离中置风险。
        """
        line_count = len(lyric_lines)
        if line_count < 2:
            return None

        token_entries = self._flatten_chunk_token_entries(chunks)
        token_count = len(token_entries)
        if token_count < line_count:
            return None

        merge_penalty = self._merge_penalty(len(lyric_lines), len(chunks))
        span_text_cache: dict[tuple[int, int], str] = {}
        span_sim_cache: dict[tuple[int, int, int], float] = {}

        def _span_text(start_idx: int, end_idx: int) -> str:
            key = (start_idx, end_idx)
            if key not in span_text_cache:
                span_text_cache[key] = self._join_token_texts(token_entries[start_idx:end_idx])
            return span_text_cache[key]

        def _span_similarity(line_idx: int, start_idx: int, end_idx: int) -> float:
            key = (line_idx, start_idx, end_idx)
            if key not in span_sim_cache:
                span_sim_cache[key] = self.text_similarity(
                    lyric_lines[line_idx].normalized_text,
                    _span_text(start_idx, end_idx),
                )
            return span_sim_cache[key]

        dp = [[float("-inf")] * (token_count + 1) for _ in range(line_count + 1)]
        back: list[list[tuple[int, int] | None]] = [[None] * (token_count + 1) for _ in range(line_count + 1)]

        for skipped in range(token_count + 1):
            dp[0][skipped] = -self.boundary_trim_penalty * skipped

        for line_idx in range(1, line_count + 1):
            remaining_lines = line_count - line_idx
            min_end = line_idx
            max_end = token_count - remaining_lines
            prefix_best_score = [float("-inf")] * (token_count + 1)
            prefix_best_end: list[int | None] = [None] * (token_count + 1)
            running_best = float("-inf")
            running_best_end: int | None = None
            for pos in range(line_idx - 1, token_count + 1):
                candidate = dp[line_idx - 1][pos] + self.token_skip_penalty * pos
                if candidate > running_best:
                    running_best = candidate
                    running_best_end = pos
                prefix_best_score[pos] = running_best
                prefix_best_end[pos] = running_best_end
            for end_idx in range(min_end, max_end + 1):
                best_score = float("-inf")
                best_prev: tuple[int, int] | None = None
                min_start = line_idx - 1
                max_start = end_idx - 1
                for start_idx in range(min_start, max_start + 1):
                    prev_end_choice = prefix_best_end[start_idx]
                    if prev_end_choice is None:
                        continue
                    prev_best = prefix_best_score[start_idx] - self.token_skip_penalty * start_idx
                    if prev_end_choice is None:
                        continue

                    similarity = _span_similarity(line_idx - 1, start_idx, end_idx)
                    span_score = 1.22 * similarity
                    if similarity < self.min_match_similarity:
                        span_score -= 0.32
                    candidate_score = prev_best + span_score
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_prev = (prev_end_choice, start_idx)

                dp[line_idx][end_idx] = best_score
                back[line_idx][end_idx] = best_prev

        best_final_score = float("-inf")
        best_final_end: int | None = None
        for end_idx in range(line_count, token_count + 1):
            base_score = dp[line_count][end_idx]
            if base_score == float("-inf"):
                continue
            trailing_skip = token_count - end_idx
            candidate_score = base_score - self.boundary_trim_penalty * trailing_skip
            if candidate_score > best_final_score:
                best_final_score = candidate_score
                best_final_end = end_idx

        if best_final_end is None:
            return None

        spans: list[tuple[int, int]] = []
        line_idx = line_count
        end_idx = best_final_end
        while line_idx > 0:
            prev = back[line_idx][end_idx]
            if prev is None:
                return None
            prev_end_idx, start_idx = prev
            spans.append((start_idx, end_idx))
            end_idx = prev_end_idx
            line_idx -= 1
        spans.reverse()

        line_assignments = []
        similarities = []
        for line_idx, (start_idx, end_idx) in enumerate(spans):
            span_entries = token_entries[start_idx:end_idx]
            if not span_entries:
                return None
            span_text = _span_text(start_idx, end_idx)
            if not span_text:
                return None
            similarity = _span_similarity(line_idx, start_idx, end_idx)
            similarities.append(similarity)
            line_assignments.append(
                {
                    "chunk_indices": self._ordered_unique([entry["segment_index"] for entry in span_entries]),
                    "chunk_text": span_text,
                    "start": float(span_entries[0]["start"]),
                    "end": float(span_entries[-1]["end"]),
                    "similarity": similarity,
                }
            )

        score = (
            best_final_score
            + 0.16 * confidence
            + 0.08 * presence
            - 0.40 * hallucination
            - 0.10 * off_center
            - merge_penalty
        )
        similarity = float(np.mean(similarities)) if similarities else 0.0
        return {
            "score": score,
            "similarity": similarity,
            "line_assignments": line_assignments,
        }

    def text_similarity(self, lyric_text: str, chunk_text: str) -> float:
        """
        计算歌词文本与转写文本的相似度。

        参数:
            lyric_text: 歌词规范化文本。
            chunk_text: 转写文本，会按歌词规范化逻辑清洗。
        """
        lyric_text = LyricLineStamp.normalize_lyric_text(lyric_text)
        chunk_text = LyricLineStamp.normalize_lyric_text(chunk_text)
        if not lyric_text or not chunk_text:
            return 0.0

        ratio = SequenceMatcher(None, lyric_text, chunk_text).ratio()
        uni_a = self._char_ngrams(lyric_text, 1)
        uni_b = self._char_ngrams(chunk_text, 1)
        bi_a = self._char_ngrams(lyric_text, 2)
        bi_b = self._char_ngrams(chunk_text, 2)

        unigram_jaccard = len(uni_a & uni_b) / max(len(uni_a | uni_b), 1)
        bigram_jaccard = len(bi_a & bi_b) / max(len(bi_a | bi_b), 1)
        containment = min(
            len(uni_a & uni_b) / max(len(uni_a), 1),
            len(uni_a & uni_b) / max(len(uni_b), 1),
        )
        score = (
            0.40 * ratio
            + 0.22 * unigram_jaccard
            + 0.26 * bigram_jaccard
            + 0.12 * containment
        )
        return float(np.clip(score, 0.0, 1.0))

    def _window_match_score(
        self,
        lyric_lines: list[LyricTokenLine],
        chunks: list[WhisperChunkResult],
    ) -> dict:
        """
        计算“若干歌词行”和“若干转写 chunk”之间的匹配分数。

        参数:
            lyric_lines: 连续歌词行列表。
            chunks: 连续转写 chunk 列表。

        说明:
            这里把多行或多 chunk 的文本分别拼接，再统一做文本相似度。
            这样可以覆盖：
            1. 单行歌词跨多个 chunk。
            2. 一个 chunk 同时覆盖多行短歌词。
        """
        lyric_text = " ".join(line.normalized_text for line in lyric_lines if line.normalized_text)
        chunk_text = " ".join(chunk.text for chunk in chunks if chunk.text)
        similarity = self.text_similarity(lyric_text, chunk_text)

        confidence = self._mean([chunk.avg_confidence for chunk in chunks]) or 0.0
        hallucination = self._mean([chunk.hallucination_risk for chunk in chunks]) or 0.0
        presence = self._mean([chunk.scores.get("vocal_presence", 0.0) for chunk in chunks]) or 0.0
        off_center = self._mean([chunk.scores.get("off_center_risk", 0.0) for chunk in chunks]) or 0.0

        line_count = len(lyric_lines)
        chunk_count = len(chunks)
        merge_penalty = self._merge_penalty(line_count, chunk_count)

        score = (
            1.55 * similarity
            + 0.16 * confidence
            + 0.08 * presence
            - 0.40 * hallucination
            - 0.10 * off_center
            - merge_penalty
        )
        if similarity < self.min_match_similarity:
            score -= 0.32

        match_info = {
            "score": score,
            "similarity": similarity,
            "line_assignments": None,
        }
        split_similarity_gate = 0.18 if line_count == 2 else 0.24
        should_try_split = (
            line_count >= 2
            and (line_count == 2 or chunk_count == 1)
            and similarity >= split_similarity_gate
        )
        if should_try_split:
            split_match = self._best_token_boundary_split(
                lyric_lines=lyric_lines,
                chunks=chunks,
                confidence=confidence,
                hallucination=hallucination,
                presence=presence,
                off_center=off_center,
            )
            if split_match is not None and split_match["score"] > match_info["score"]:
                match_info = split_match
        return match_info

    @staticmethod
    def _distribute_line_times(
        lines: list[LyricTokenLine],
        start: float | None,
        end: float | None,
    ) -> list[tuple[float | None, float | None]]:
        """
        将一个 chunk 覆盖多行歌词时，把时间区间按字符长度比例切分。

        参数:
            lines: 连续歌词行。
            start: 总开始时间。
            end: 总结束时间。
        """
        if start is None or end is None or end <= start:
            return [(None, None) for _ in lines]

        weights = [max(len(line.normalized_text or line.text or ""), 1) for line in lines]
        total = sum(weights)
        durations = []
        current = start
        total_span = end - start

        for idx, weight in enumerate(weights):
            if idx == len(weights) - 1:
                line_end = end
            else:
                line_end = current + total_span * (weight / max(total, 1))
            durations.append((current, line_end))
            current = line_end
        return durations

    def align(self, lyric: LyricLineStamp, transcription: WhisperTrackResult | dict) -> LyricAlignmentResult:
        """
        对歌词和转写结果执行单调约束对齐。

        参数:
            lyric: 规范化后的歌词对象。
            transcription: 转写结果对象或字典。
        """
        music_path, chunk_list = self._coerce_chunks(transcription)
        lyric_lines = lyric.lyric_lines
        if not lyric_lines:
            return LyricAlignmentResult(music_path=music_path, items=[])

        n = len(lyric_lines)
        m = len(chunk_list)
        dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] - self.skip_line_penalty
            back[i][0] = ("skip_line", i - 1, 0, 1, 0, 0.0, 0.0)
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] - self.skip_chunk_penalty
            back[0][j] = ("skip_chunk", 0, j - 1, 0, 1, 0.0, 0.0)

        for i in range(n + 1):
            for j in range(m + 1):
                current = dp[i][j]
                if current == float("-inf"):
                    continue

                if i < n and current - self.skip_line_penalty > dp[i + 1][j]:
                    dp[i + 1][j] = current - self.skip_line_penalty
                    back[i + 1][j] = ("skip_line", i, j, 1, 0, 0.0, 0.0)

                if j < m and current - self.skip_chunk_penalty > dp[i][j + 1]:
                    dp[i][j + 1] = current - self.skip_chunk_penalty
                    back[i][j + 1] = ("skip_chunk", i, j, 0, 1, 0.0, 0.0)

                max_line_span = min(self.max_lines_per_chunk, n - i)
                max_chunk_span = min(self.max_chunks_per_line, m - j)
                for line_span in range(1, max_line_span + 1):
                    for chunk_span in range(1, max_chunk_span + 1):
                        lyric_window = lyric_lines[i:i + line_span]
                        chunk_window = chunk_list[j:j + chunk_span]
                        match_info = self._window_match_score(lyric_window, chunk_window)
                        next_score = current + match_info["score"]
                        if next_score > dp[i + line_span][j + chunk_span]:
                            dp[i + line_span][j + chunk_span] = next_score
                            back[i + line_span][j + chunk_span] = (
                                "match",
                                i,
                                j,
                                line_span,
                                chunk_span,
                                match_info,
                            )

        assignments: list[dict | None] = [None] * n
        i, j = n, m
        while i > 0 or j > 0:
            step = back[i][j]
            if step is None:
                break
            action, prev_i, prev_j, line_span, chunk_span, *payload = step
            if action == "match":
                match_info = payload[0]
                lines = lyric_lines[prev_i:i]
                chunks = chunk_list[prev_j:j]
                chunk_indices = [chunk.segment_index for chunk in chunks]
                chunk_text = " ".join(chunk.text for chunk in chunks if chunk.text).strip() or None
                start = chunks[0].start if chunks else None
                end = chunks[-1].end if chunks else None
                confidence = self._mean([chunk.avg_confidence for chunk in chunks])
                hallucination = self._mean([chunk.hallucination_risk for chunk in chunks])
                match_value = float(match_info["score"])
                similarity = float(match_info["similarity"])

                if match_info.get("line_assignments"):
                    for offset, line_assignment in enumerate(match_info["line_assignments"]):
                        assignments[prev_i + offset] = {
                            "chunk_indices": line_assignment["chunk_indices"],
                            "chunk_text": line_assignment["chunk_text"],
                            "start": line_assignment["start"],
                            "end": line_assignment["end"],
                            "similarity": line_assignment["similarity"],
                            "score": match_value / max(line_span, 1),
                            "confidence": confidence,
                            "hallucination_risk": hallucination,
                        }
                elif line_span == 1:
                    assignments[prev_i] = {
                        "chunk_indices": chunk_indices,
                        "chunk_text": chunk_text,
                        "start": start,
                        "end": end,
                        "similarity": similarity,
                        "score": match_value,
                        "confidence": confidence,
                        "hallucination_risk": hallucination,
                    }
                else:
                    line_times = self._distribute_line_times(lines, start, end)
                    for offset, (line_start, line_end) in enumerate(line_times):
                        assignments[prev_i + offset] = {
                            "chunk_indices": chunk_indices,
                            "chunk_text": chunk_text,
                            "start": line_start,
                            "end": line_end,
                            "similarity": similarity,
                            "score": match_value / max(line_span, 1),
                            "confidence": confidence,
                            "hallucination_risk": hallucination,
                        }
            i, j = prev_i, prev_j

        items: list[AlignedLyricLine] = []
        for idx, line in enumerate(lyric_lines):
            assigned = assignments[idx]
            if assigned is None:
                items.append(
                    AlignedLyricLine(
                        line_index=idx,
                        line=line,
                        normalized_text=line.normalized_text,
                        chunk_indices=[],
                        chunk_text=None,
                        start=None,
                        end=None,
                        similarity=0.0,
                        score=0.0,
                        confidence=None,
                        hallucination_risk=None,
                    )
                )
                continue

            items.append(
                AlignedLyricLine(
                    line_index=idx,
                    line=line,
                    normalized_text=line.normalized_text,
                    chunk_indices=assigned["chunk_indices"],
                    chunk_text=assigned["chunk_text"],
                    start=assigned["start"],
                    end=assigned["end"],
                    similarity=assigned["similarity"],
                    score=assigned["score"],
                    confidence=assigned["confidence"],
                    hallucination_risk=assigned["hallucination_risk"],
                )
            )

        return LyricAlignmentResult(music_path=music_path, items=items)

    def refine_with_lyric_timestamps(
        self,
        lyric: LyricLineStamp,
        alignment_result: LyricAlignmentResult,
        transcription: WhisperTrackResult | dict,
    ) -> LyricAlignmentResult:
        """
        用原歌词时间轴对 DP 对齐结果做稳健后处理。

        说明:
            仅在歌词本身包含真实时间戳时启用。它会：
            1. 从 DP 结果中选取较可靠的 matched 行作为锚点。
            2. 估计全局或分段 offset。
            3. 将 DP start 与 `lyric_timestamp + offset` 融合。
            4. 再做一次单调平滑，减少局部错词对起始时间的污染。
        """
        if not getattr(lyric, "has_real_timestamps", True):
            return alignment_result

        from Lazulite.OffsetAlign import OffsetAligner

        _, chunk_list = self._coerce_chunks(transcription)
        offset_helper = OffsetAligner()
        section_to_merged, merged_sections = offset_helper._merge_sections(chunk_list)
        chunk_section_by_segment = {
            int(chunk.segment_index): chunk.section_index
            for chunk in chunk_list
        }

        anchor_items = []
        offset_candidates: list[_HybridOffsetAnchor] = []

        for item in alignment_result.items:
            if item.start is None or item.line.timestamp is None:
                continue
            reliability = self._item_alignment_reliability(item)
            if reliability < 0.24:
                continue
            offset = float(item.start - item.line.timestamp)
            section_index = self._infer_item_section_index(item, chunk_section_by_segment)
            merged_section_index = section_to_merged.get(section_index)
            offset_candidates.append(
                _HybridOffsetAnchor(
                    line_index=item.line_index,
                    merged_section_index=merged_section_index,
                    delta=offset,
                    score=reliability,
                    chunk_confidence=item.confidence,
                )
            )
            anchor_items.append(
                {
                    "line_index": item.line_index,
                    "offset": offset,
                    "reliability": reliability,
                    "section_index": section_index,
                    "merged_section_index": merged_section_index,
                }
            )

        global_cluster = offset_helper._largest_consistent_cluster(offset_candidates)
        section_estimates, reliable_anchors = offset_helper._estimate_section_offsets(global_cluster, merged_sections)
        is_reliable = bool(section_estimates) and len(reliable_anchors) >= offset_helper.min_anchor_count

        if not is_reliable and len(global_cluster) >= offset_helper.min_anchor_count:
            deltas = [item.delta for item in global_cluster]
            global_offset = offset_helper._median(deltas)
            mad = offset_helper._median([abs(delta - global_offset) for delta in deltas])
            if mad <= offset_helper.max_section_mad:
                line_indices = sorted(item.line_index for item in global_cluster)
                section_estimates = [{
                    "section_index": "hybrid_forced_global",
                    "source_section_indices": [item["merged_section_index"] for item in merged_sections],
                    "offset": global_offset,
                    "anchor_count": len(global_cluster),
                    "mad": mad,
                    "line_start": line_indices[0],
                    "line_end": line_indices[-1],
                    "confidence": offset_helper._mean([item.chunk_confidence for item in global_cluster]),
                }]
                reliable_anchors = global_cluster
                is_reliable = True

        if not is_reliable:
            return alignment_result

        line_offsets = offset_helper._assign_line_offsets(len(alignment_result.items), section_estimates)
        if not line_offsets or all(offset is None for offset in line_offsets):
            return alignment_result

        global_offset = float(section_estimates[0]["offset"])
        section_offsets = {
            str(section.get("section_index")): float(section["offset"])
            for section in section_estimates
            if section.get("offset") is not None
        }

        refined = LyricAlignmentResult(
            music_path=alignment_result.music_path,
            items=[
                AlignedLyricLine(
                    line_index=item.line_index,
                    line=item.line,
                    normalized_text=item.normalized_text,
                    chunk_indices=list(item.chunk_indices),
                    chunk_text=item.chunk_text,
                    start=item.start,
                    end=item.end,
                    similarity=item.similarity,
                    score=item.score,
                    confidence=item.confidence,
                    hallucination_risk=item.hallucination_risk,
                )
                for item in alignment_result.items
            ],
            strategy=alignment_result.strategy,
            details=dict(alignment_result.details or {}),
        )

        refined_starts: list[float | None] = []
        duration_hints: list[float | None] = []

        for item in refined.items:
            duration = None
            if item.start is not None and item.end is not None:
                duration = max(0.0, float(item.end - item.start))
            duration_hints.append(duration)

            if item.line.timestamp is None:
                refined_starts.append(None if item.start is None else float(item.start))
                continue

            line_offset = line_offsets[item.line_index]
            if line_offset is None:
                refined_starts.append(None if item.start is None else float(item.start))
                continue
            prior = max(0.0, float(item.line.timestamp + float(line_offset)))
            refined_starts.append(prior)

        refined_starts = self._smooth_monotonic_timestamps(refined_starts)

        next_known_start = None
        for idx in range(len(refined.items) - 1, -1, -1):
            item = refined.items[idx]
            refined_start = refined_starts[idx]
            item.start = refined_start
            duration = duration_hints[idx]

            if refined_start is None:
                item.end = None if duration is None else item.end
                continue

            end = refined_start if duration is None else float(refined_start + duration)
            if next_known_start is not None:
                end = min(end, float(next_known_start))
            item.end = max(float(refined_start), float(end))
            next_known_start = refined_start

        refined.details["hybrid_refinement"] = {
            "enabled": True,
            "anchor_count": len(anchor_items),
            "global_offset": float(global_offset),
            "merged_sections": merged_sections,
            "sections": section_estimates,
            "section_offsets": section_offsets,
        }
        return refined
