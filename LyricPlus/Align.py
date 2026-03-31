from difflib import SequenceMatcher

import numpy as np

from LyricPlus.Lyric import LyricLineStamp, LyricTokenLine
from LyricPlus.Transcribe import WhisperChunkResult, WhisperTrackResult


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

    def __init__(self, music_path: str, items: list[AlignedLyricLine]):
        """
        初始化整首歌对齐结果。

        参数:
            music_path: 原始音频路径。
            items: 行级对齐结果列表。
        """
        self.music_path = music_path
        self.items = items

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
        }

    def to_dict(self) -> dict:
        """
        将整首歌对齐结果转换为字典。
        """
        return {
            "music_path": self.music_path,
            "stats": self.stats,
            "items": [item.to_dict() for item in self.items],
        }


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
        max_lines_per_chunk: int = 2,
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
                    start=item["start"],
                    end=item["end"],
                    core_start=item["core_start"],
                    core_end=item["core_end"],
                    text=item.get("text", ""),
                    language=item.get("language"),
                    avg_logprob=item.get("avg_logprob"),
                    avg_confidence=item.get("avg_confidence"),
                    confidence_source=item.get("confidence_source", ""),
                    token_confidences=item.get("token_confidences"),
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
    ) -> tuple[float, float]:
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
        merge_penalty = 0.0
        if line_count > 1:
            merge_penalty += 0.05 * (line_count - 1)
        if chunk_count > 1:
            merge_penalty += 0.04 * (chunk_count - 1)

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
        return score, similarity

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
                        match_value, similarity = self._window_match_score(lyric_window, chunk_window)
                        next_score = current + match_value
                        if next_score > dp[i + line_span][j + chunk_span]:
                            dp[i + line_span][j + chunk_span] = next_score
                            back[i + line_span][j + chunk_span] = (
                                "match",
                                i,
                                j,
                                line_span,
                                chunk_span,
                                match_value,
                                similarity,
                            )

        assignments: list[dict | None] = [None] * n
        i, j = n, m
        while i > 0 or j > 0:
            step = back[i][j]
            if step is None:
                break
            action, prev_i, prev_j, line_span, chunk_span, match_value, similarity = step
            if action == "match":
                lines = lyric_lines[prev_i:i]
                chunks = chunk_list[prev_j:j]
                chunk_indices = [chunk.segment_index for chunk in chunks]
                chunk_text = " ".join(chunk.text for chunk in chunks if chunk.text).strip() or None
                start = chunks[0].start if chunks else None
                end = chunks[-1].end if chunks else None
                confidence = self._mean([chunk.avg_confidence for chunk in chunks])
                hallucination = self._mean([chunk.hallucination_risk for chunk in chunks])

                if line_span == 1:
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
