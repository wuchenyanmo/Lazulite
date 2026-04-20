from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import numpy as np

from Lazulite.Align import AlignedLyricLine, LyricAlignmentResult, LyricAligner
from Lazulite.Lyric import LyricLineStamp, LyricTokenLine
from Lazulite.Transcribe import WhisperChunkResult, WhisperTrackResult


@dataclass
class OffsetAnchor:
    line_index: int
    segment_index: int
    section_index: int | None
    merged_section_index: int | None
    anchor_time: float
    lyric_timestamp: float
    delta: float
    prefix_similarity: float
    full_similarity: float
    chunk_confidence: float | None
    score: float
    chunk_text: str

    def to_dict(self) -> dict:
        return {
            "line_index": self.line_index,
            "segment_index": self.segment_index,
            "section_index": self.section_index,
            "merged_section_index": self.merged_section_index,
            "anchor_time": self.anchor_time,
            "lyric_timestamp": self.lyric_timestamp,
            "delta": self.delta,
            "prefix_similarity": self.prefix_similarity,
            "full_similarity": self.full_similarity,
            "chunk_confidence": self.chunk_confidence,
            "score": self.score,
            "chunk_text": self.chunk_text,
        }


class OffsetAligner:
    """
    基于高质量 anchor 的分段偏移对齐器。
    """

    def __init__(
        self,
        min_prefix_similarity: float = 0.68,
        min_full_similarity: float = 0.50,
        min_anchor_count: int = 3,
        min_section_anchor_count: int = 2,
        min_merged_section_chunks: int = 5,
        max_section_merge_gap_sec: float = 3.0,
        max_delta_spread: float = 1.0,
        max_section_mad: float = 0.35,
        min_length_ratio: float = 0.55,
        max_length_ratio: float = 1.8,
        prefix_chars: int = 8,
    ):
        self.min_prefix_similarity = min_prefix_similarity
        self.min_full_similarity = min_full_similarity
        self.min_anchor_count = min_anchor_count
        self.min_section_anchor_count = min_section_anchor_count
        self.min_merged_section_chunks = min_merged_section_chunks
        self.max_section_merge_gap_sec = max_section_merge_gap_sec
        self.max_delta_spread = max_delta_spread
        self.max_section_mad = max_section_mad
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.prefix_chars = prefix_chars

    @staticmethod
    def _coerce_chunks(transcription: WhisperTrackResult | dict) -> tuple[str, list[WhisperChunkResult]]:
        return LyricAligner._coerce_chunks(transcription)

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return float(np.mean(clean)) if clean else None

    @staticmethod
    def _median(values: list[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _prefix_similarity(a: str, b: str, prefix_chars: int) -> float:
        a = LyricLineStamp.normalize_lyric_text(a)
        b = LyricLineStamp.normalize_lyric_text(b)
        if not a or not b:
            return 0.0
        usable = min(len(a), len(b), max(prefix_chars, 1))
        return float(SequenceMatcher(None, a[:usable], b[:usable]).ratio())

    @staticmethod
    def _line_match_text(line: LyricTokenLine) -> str:
        text = line.normalized_text or line.text or ""
        text = re.split(r"[【\[\(（「『]", text, maxsplit=1)[0]
        return LyricLineStamp.normalize_lyric_text(text)

    def _chunk_is_eligible(self, chunk: WhisperChunkResult) -> bool:
        if not chunk.text:
            return False
        if float(chunk.avg_confidence or 0.0) < 0.42:
            return False
        if float(chunk.hallucination_risk) > 0.55:
            return False
        if chunk.min_token_confidence is not None and float(chunk.min_token_confidence) < 0.06:
            return False
        if chunk.low_conf_token_ratio is not None and float(chunk.low_conf_token_ratio) > 0.70:
            return False
        if float(chunk.self_repeat_score) > 0.60:
            return False
        if float(chunk.scores.get("vocal_presence", 0.0)) < 0.28:
            return False
        return True

    @staticmethod
    def _anchor_time(chunk: WhisperChunkResult) -> float:
        tokens = chunk.tokens or []
        for token in tokens:
            text = str(token.get("text", token.get("token", "")))
            if text.startswith("<|") and text.endswith("|>"):
                continue
            if not text.strip():
                continue
            start = token.get("start")
            if start is not None:
                return float(chunk.start + float(start))
        if chunk.core_start is not None:
            return float(chunk.core_start)
        return float(chunk.start)

    def _merge_sections(self, chunks: list[WhisperChunkResult]) -> tuple[dict[int | None, int | None], list[dict]]:
        """
        将过碎的原始 section 合并成更粗的 offset section。

        参数:
            chunks: 当前用于 offset 的转写 chunk 列表。
        """
        if not chunks:
            return {}, []

        per_section = []
        current = None
        for chunk in sorted(chunks, key=lambda item: item.segment_index):
            section_index = chunk.section_index
            if current is None or current["section_index"] != section_index:
                current = {
                    "section_index": section_index,
                    "chunk_indices": [chunk.segment_index],
                    "chunk_count": 1,
                    "start": float(chunk.start),
                    "end": float(chunk.end),
                }
                per_section.append(current)
                continue
            current["chunk_indices"].append(chunk.segment_index)
            current["chunk_count"] += 1
            current["end"] = float(chunk.end)

        merged_sections = []
        current = None
        for section in per_section:
            if current is None:
                current = {
                    "merged_section_index": 1,
                    "section_indices": [section["section_index"]],
                    "chunk_indices": list(section["chunk_indices"]),
                    "chunk_count": int(section["chunk_count"]),
                    "start": float(section["start"]),
                    "end": float(section["end"]),
                }
                continue

            gap = float(section["start"] - current["end"])
            force_split = (
                current["chunk_count"] >= self.min_merged_section_chunks
                and gap > self.max_section_merge_gap_sec
            )
            if force_split:
                merged_sections.append(current)
                current = {
                    "merged_section_index": len(merged_sections) + 1,
                    "section_indices": [section["section_index"]],
                    "chunk_indices": list(section["chunk_indices"]),
                    "chunk_count": int(section["chunk_count"]),
                    "start": float(section["start"]),
                    "end": float(section["end"]),
                }
                continue

            current["section_indices"].append(section["section_index"])
            current["chunk_indices"].extend(section["chunk_indices"])
            current["chunk_count"] += int(section["chunk_count"])
            current["end"] = float(section["end"])

        if current is not None:
            merged_sections.append(current)

        if len(merged_sections) >= 2 and merged_sections[-1]["chunk_count"] < self.min_merged_section_chunks:
            tail = merged_sections.pop()
            merged_sections[-1]["section_indices"].extend(tail["section_indices"])
            merged_sections[-1]["chunk_indices"].extend(tail["chunk_indices"])
            merged_sections[-1]["chunk_count"] += int(tail["chunk_count"])
            merged_sections[-1]["end"] = float(tail["end"])

        section_to_merged = {}
        for merged in merged_sections:
            for section_index in merged["section_indices"]:
                section_to_merged[section_index] = merged["merged_section_index"]
        return section_to_merged, merged_sections

    def _build_anchor_candidates(
        self,
        lyric_lines: list[LyricTokenLine],
        chunks: list[WhisperChunkResult],
        section_to_merged: dict[int | None, int | None],
    ) -> list[OffsetAnchor]:
        candidates: list[OffsetAnchor] = []
        text_aligner = LyricAligner()
        for chunk in chunks:
            if not self._chunk_is_eligible(chunk):
                continue
            chunk_text = LyricLineStamp.normalize_lyric_text(chunk.text)
            if not chunk_text:
                continue
            for line_index, line in enumerate(lyric_lines):
                lyric_text = self._line_match_text(line)
                if not lyric_text:
                    continue
                length_ratio = len(chunk_text) / max(len(lyric_text), 1)
                if length_ratio < self.min_length_ratio or length_ratio > self.max_length_ratio:
                    continue
                full_similarity = text_aligner.text_similarity(lyric_text=lyric_text, chunk_text=chunk_text)
                prefix_similarity = self._prefix_similarity(lyric_text, chunk_text, self.prefix_chars)
                if prefix_similarity < self.min_prefix_similarity or full_similarity < self.min_full_similarity:
                    continue
                anchor_time = self._anchor_time(chunk)
                score = 1.30 * prefix_similarity + 1.00 * full_similarity
                candidates.append(
                    OffsetAnchor(
                        line_index=line_index,
                        segment_index=chunk.segment_index,
                        section_index=chunk.section_index,
                        merged_section_index=section_to_merged.get(chunk.section_index),
                        anchor_time=anchor_time,
                        lyric_timestamp=float(line.timestamp),
                        delta=float(anchor_time - line.timestamp),
                        prefix_similarity=prefix_similarity,
                        full_similarity=full_similarity,
                        chunk_confidence=chunk.avg_confidence,
                        score=score,
                        chunk_text=chunk.text,
                    )
                )
        return candidates

    def _select_monotonic_anchors(self, candidates: list[OffsetAnchor]) -> list[OffsetAnchor]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: (item.segment_index, item.line_index, -item.score))
        dp = [anchor.score for anchor in ordered]
        prev = [-1] * len(ordered)
        for idx, anchor in enumerate(ordered):
            for prev_idx in range(idx):
                prev_anchor = ordered[prev_idx]
                if prev_anchor.segment_index >= anchor.segment_index:
                    continue
                if prev_anchor.line_index >= anchor.line_index:
                    continue
                candidate_score = dp[prev_idx] + anchor.score
                if candidate_score > dp[idx]:
                    dp[idx] = candidate_score
                    prev[idx] = prev_idx

        best_idx = max(range(len(ordered)), key=lambda idx: dp[idx])
        chain = []
        while best_idx >= 0:
            chain.append(ordered[best_idx])
            best_idx = prev[best_idx]
        chain.reverse()
        return chain

    def _largest_consistent_cluster(self, anchors: list[OffsetAnchor]) -> list[OffsetAnchor]:
        if not anchors:
            return []
        ordered = sorted(anchors, key=lambda item: item.delta)
        best = ordered[:1]
        left = 0
        for right in range(len(ordered)):
            while ordered[right].delta - ordered[left].delta > self.max_delta_spread:
                left += 1
            window = ordered[left:right + 1]
            if len(window) > len(best):
                best = window
            elif len(window) == len(best) and len(window) > 0:
                if sum(item.score for item in window) > sum(item.score for item in best):
                    best = window
        return best

    def _estimate_section_offsets(self, anchors: list[OffsetAnchor], merged_sections: list[dict]) -> tuple[list[dict], list[OffsetAnchor]]:
        by_section: dict[int | None, list[OffsetAnchor]] = {}
        for anchor in anchors:
            by_section.setdefault(anchor.merged_section_index, []).append(anchor)

        merged_lookup = {item["merged_section_index"]: item for item in merged_sections}

        section_estimates = []
        reliable_anchors: list[OffsetAnchor] = []
        for section_index, section_anchors in sorted(by_section.items(), key=lambda item: (-1 if item[0] is None else item[0])):
            cluster = self._largest_consistent_cluster(section_anchors)
            if len(cluster) < self.min_section_anchor_count:
                continue
            deltas = [item.delta for item in cluster]
            offset = self._median(deltas)
            mad = self._median([abs(delta - offset) for delta in deltas])
            if mad > self.max_section_mad:
                continue
            line_indices = sorted(item.line_index for item in cluster)
            section_estimates.append(
                {
                    "section_index": section_index,
                    "source_section_indices": [] if merged_lookup.get(section_index) is None else merged_lookup[section_index]["section_indices"],
                    "offset": offset,
                    "anchor_count": len(cluster),
                    "mad": mad,
                    "line_start": line_indices[0],
                    "line_end": line_indices[-1],
                    "confidence": self._mean([item.chunk_confidence for item in cluster]),
                }
            )
            reliable_anchors.extend(cluster)

        section_estimates.sort(key=lambda item: item["line_start"])
        return section_estimates, reliable_anchors

    @staticmethod
    def _assign_line_offsets(line_count: int, sections: list[dict]) -> list[float | None]:
        if not sections:
            return [None] * line_count
        offsets: list[float | None] = [None] * line_count
        boundaries = []
        for idx in range(len(sections) - 1):
            current = sections[idx]
            nxt = sections[idx + 1]
            boundaries.append((current["line_end"] + nxt["line_start"]) // 2)

        start = 0
        for idx, section in enumerate(sections):
            end = line_count - 1 if idx == len(sections) - 1 else boundaries[idx]
            for line_index in range(start, end + 1):
                offsets[line_index] = float(section["offset"])
            start = end + 1
        return offsets

    @staticmethod
    def _build_items_from_line_offsets(
        lyric_lines: list[LyricTokenLine],
        line_offsets: list[float | None],
        reliable_anchors: list[OffsetAnchor],
    ) -> list[AlignedLyricLine]:
        anchor_by_line = {anchor.line_index: anchor for anchor in reliable_anchors}
        items: list[AlignedLyricLine] = []
        for idx, line in enumerate(lyric_lines):
            offset = line_offsets[idx]
            start = None if offset is None else float(max(0.0, line.timestamp + offset))
            end = None
            if start is not None and idx + 1 < len(lyric_lines) and line_offsets[idx + 1] is not None:
                end = float(max(start, lyric_lines[idx + 1].timestamp + float(line_offsets[idx + 1])))
            anchor = anchor_by_line.get(idx)
            items.append(
                AlignedLyricLine(
                    line_index=idx,
                    line=line,
                    normalized_text=line.normalized_text,
                    chunk_indices=[] if anchor is None else [anchor.segment_index],
                    chunk_text=None if anchor is None else anchor.chunk_text,
                    start=start,
                    end=end,
                    similarity=0.0 if anchor is None else anchor.full_similarity,
                    score=0.0 if anchor is None else anchor.score,
                    confidence=None if anchor is None else anchor.chunk_confidence,
                    hallucination_risk=None,
                )
            )
        return items

    @staticmethod
    def _build_original_lyric_items(lyric_lines: list[LyricTokenLine]) -> list[AlignedLyricLine]:
        items: list[AlignedLyricLine] = []
        for idx, line in enumerate(lyric_lines):
            start = float(max(0.0, line.timestamp))
            end = None if idx + 1 >= len(lyric_lines) else float(max(start, lyric_lines[idx + 1].timestamp))
            items.append(
                AlignedLyricLine(
                    line_index=idx,
                    line=line,
                    normalized_text=line.normalized_text,
                    chunk_indices=[],
                    chunk_text=None,
                    start=start,
                    end=end,
                    similarity=0.0,
                    score=0.0,
                    confidence=None,
                    hallucination_risk=None,
                )
            )
        return items

    def align(
        self,
        lyric: LyricLineStamp,
        transcription: WhisperTrackResult | dict,
        force_global_fallback: bool = False,
        fallback_to_original_on_no_anchor: bool = False,
    ) -> LyricAlignmentResult:
        music_path, chunks = self._coerce_chunks(transcription)
        lyric_lines = lyric.lyric_lines
        if not lyric_lines:
            return LyricAlignmentResult(music_path=music_path, items=[], strategy="offset")
        if not getattr(lyric, "has_real_timestamps", True):
            return LyricAlignmentResult(
                music_path=music_path,
                items=[],
                strategy="offset",
                details={
                    "is_reliable": False,
                    "fallback_reason": "missing_lyric_timestamps",
                    "candidate_count": 0,
                    "anchor_count": 0,
                    "cluster_anchor_count": 0,
                    "reliable_anchor_count": 0,
                    "merged_sections": [],
                    "sections": [],
                    "anchors": [],
                },
            )

        section_to_merged, merged_sections = self._merge_sections(chunks)
        candidates = self._build_anchor_candidates(
            lyric_lines=lyric_lines,
            chunks=chunks,
            section_to_merged=section_to_merged,
        )
        anchors = self._select_monotonic_anchors(candidates)
        global_cluster = self._largest_consistent_cluster(anchors)
        section_estimates, reliable_anchors = self._estimate_section_offsets(global_cluster, merged_sections)

        is_reliable = bool(section_estimates) and len(reliable_anchors) >= self.min_anchor_count
        if not is_reliable and len(global_cluster) >= self.min_anchor_count:
            deltas = [item.delta for item in global_cluster]
            offset = self._median(deltas)
            mad = self._median([abs(delta - offset) for delta in deltas])
            if mad <= self.max_section_mad:
                line_indices = sorted(item.line_index for item in global_cluster)
                section_estimates = [{
                    "section_index": None,
                    "offset": offset,
                    "anchor_count": len(global_cluster),
                    "mad": mad,
                    "line_start": line_indices[0],
                    "line_end": line_indices[-1],
                    "confidence": self._mean([item.chunk_confidence for item in global_cluster]),
                }]
                reliable_anchors = global_cluster
                is_reliable = True

        details = {
            "is_reliable": is_reliable,
            "candidate_count": len(candidates),
            "anchor_count": len(anchors),
            "cluster_anchor_count": len(global_cluster),
            "reliable_anchor_count": len(reliable_anchors),
            "merged_sections": merged_sections,
            "sections": section_estimates,
            "anchors": [anchor.to_dict() for anchor in reliable_anchors],
        }

        if not is_reliable and force_global_fallback:
            if anchors:
                deltas = [item.delta for item in anchors]
                offset = self._median(deltas)
                mad = self._median([abs(delta - offset) for delta in deltas])
                section_estimates = [{
                    "section_index": "forced_global",
                    "source_section_indices": [item["merged_section_index"] for item in merged_sections],
                    "offset": offset,
                    "anchor_count": len(anchors),
                    "mad": mad,
                    "line_start": 0,
                    "line_end": len(lyric_lines) - 1,
                    "confidence": self._mean([item.chunk_confidence for item in anchors]),
                }]
                reliable_anchors = anchors
                line_offsets = self._assign_line_offsets(len(lyric_lines), section_estimates)
                details["forced_global_fallback"] = True
                details["fallback_reason"] = "insufficient_or_inconsistent_anchors"
                details["sections"] = section_estimates
                details["anchors"] = [anchor.to_dict() for anchor in reliable_anchors]
                items = self._build_items_from_line_offsets(lyric_lines, line_offsets, reliable_anchors)
                return LyricAlignmentResult(
                    music_path=music_path,
                    items=items,
                    strategy="offset",
                    details=details,
                )

            if fallback_to_original_on_no_anchor:
                details["fallback_to_original_lyric"] = True
                details["fallback_reason"] = "no_anchor"
                items = self._build_original_lyric_items(lyric_lines)
                return LyricAlignmentResult(
                    music_path=music_path,
                    items=items,
                    strategy="offset",
                    details=details,
                )

        line_offsets = self._assign_line_offsets(len(lyric_lines), section_estimates) if is_reliable else [None] * len(lyric_lines)
        items = self._build_items_from_line_offsets(lyric_lines, line_offsets, reliable_anchors)
        return LyricAlignmentResult(
            music_path=music_path,
            items=items,
            strategy="offset",
            details=details,
        )
