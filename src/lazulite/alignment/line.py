"""基于单调动态规划的行级歌词对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lazulite.lyrics.similarity import text_similarity
from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult, TranscriptionSegment


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """一次对齐使用的只读视图，不持有模型、网络或历史状态。"""

    segment: VocalSegment
    transcription: TranscriptionSegment


@dataclass(frozen=True, slots=True)
class LineAlignmentConfig:
    minimum_similarity: float = 0.08
    skip_line_penalty: float = 0.34
    skip_evidence_penalty: float = 0.24
    max_evidence_per_line: int = 2
    max_lines_per_evidence: int = 4

    def __post_init__(self) -> None:
        if self.max_evidence_per_line < 1 or self.max_lines_per_evidence < 1:
            raise ValueError("每次匹配的行数和分片数必须为正数")


def _evidence(analysis: VocalAnalysis, transcription: TranscriptionResult) -> tuple[AlignmentEvidence, ...]:
    by_id = {segment.id: segment for segment in analysis.segments}
    return tuple(
        AlignmentEvidence(by_id[item.source_segment_id], item)
        for item in transcription.segments
        if item.source_segment_id in by_id and item.text.strip()
    )


def _merge_penalty(line_count: int, evidence_count: int) -> float:
    return 0.05 * max(0, line_count - 1) + 0.04 * max(0, evidence_count - 1)


def _score(lines: tuple[str, ...], evidence: tuple[AlignmentEvidence, ...]) -> float:
    value = 1.55 * text_similarity("\n".join(lines), "\n".join(item.transcription.text for item in evidence))
    presence = [item.segment.presence for item in evidence if item.segment.presence is not None]
    risk = [item.segment.off_center_risk for item in evidence if item.segment.off_center_risk is not None]
    if presence:
        value += 0.08 * sum(presence) / len(presence)
    if risk:
        value -= 0.10 * sum(risk) / len(risk)
    return value - _merge_penalty(len(lines), len(evidence))


def _timing(evidence: tuple[AlignmentEvidence, ...]) -> tuple[float | None, float | None, Literal["model_units", "acoustic_core"], tuple[str, ...]]:
    units = tuple(unit for item in evidence for unit in item.transcription.timed_text)
    starts = [unit.start for unit in units if unit.start is not None]
    ends = [unit.end for unit in units if unit.end is not None]
    if starts:
        return min(starts), max(ends) if ends else None, "model_units", ()
    issues = ["coarse_timing"]
    if any(item.transcription.confidence is None for item in evidence):
        issues.append("missing_model_confidence")
    return min(item.segment.core_start for item in evidence), max(item.segment.core_end for item in evidence), "acoustic_core", tuple(issues)


def _interpolated_timing(
    lines: tuple[str, ...], evidence: tuple[AlignmentEvidence, ...], line_index: int,
) -> tuple[float, float, tuple[str, ...]]:
    """多行共用声学窗口时只做粗分配，明确不挪用 token 时间。"""
    start = min(item.segment.core_start for item in evidence)
    end = max(item.segment.core_end for item in evidence)
    weights = [max(len(line), 1) for line in lines]
    total = sum(weights)
    leading = sum(weights[:line_index]) / total
    trailing = sum(weights[:line_index + 1]) / total
    return start + (end - start) * leading, start + (end - start) * trailing, ("coarse_timing", "merged_line_evidence")


ActionKind = Literal["line", "evidence", "match"]
Action = tuple[ActionKind, int, int]


def align_lines(document: LyricDocument, transcription: TranscriptionResult, analysis: VocalAnalysis, *, config: LineAlignmentConfig = LineAlignmentConfig()) -> LineAlignmentResult:
    """保留所有输入行，并在受限范围内匹配多行与多分片。"""
    evidence = _evidence(analysis, transcription)
    rows, columns = len(document.lines), len(evidence)
    dp = [[float("-inf")] * (columns + 1) for _ in range(rows + 1)]
    step: list[list[Action | None]] = [[None] * (columns + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0

    def update(row: int, column: int, value: float, action: Action) -> None:
        if value > dp[row][column]:
            dp[row][column] = value
            step[row][column] = action

    for row in range(rows + 1):
        for column in range(columns + 1):
            current = dp[row][column]
            if current == float("-inf"):
                continue
            if row < rows:
                update(row + 1, column, current - config.skip_line_penalty, ("line", 1, 0))
            if column < columns:
                update(row, column + 1, current - config.skip_evidence_penalty, ("evidence", 0, 1))
            for line_count in range(1, min(config.max_lines_per_evidence, rows - row) + 1):
                lines = tuple(item.text for item in document.lines[row:row + line_count])
                for evidence_count in range(1, min(config.max_evidence_per_line, columns - column) + 1):
                    group = evidence[column:column + evidence_count]
                    similarity = text_similarity("\n".join(lines), "\n".join(item.transcription.text for item in group))
                    if similarity >= config.minimum_similarity:
                        update(row + line_count, column + evidence_count, current + _score(lines, group), ("match", line_count, evidence_count))

    matches: dict[int, tuple[int, int]] = {}
    row, column = rows, columns
    while row or column:
        action = step[row][column]
        if action is None:
            break
        kind, line_count, evidence_count = action
        if kind == "match":
            for index in range(row - line_count, row):
                matches[index] = (column - evidence_count, column)
            row -= line_count
            column -= evidence_count
        elif kind == "line":
            row -= 1
        else:
            column -= 1

    output: list[AlignedLyricLine] = []
    for index, line in enumerate(document.lines):
        matched_range = matches.get(index)
        if matched_range is None:
            output.append(AlignedLyricLine(line.id, None, None, 0.0, None, "unresolved", (), ("low_text_match",)))
            continue
        group = evidence[matched_range[0]:matched_range[1]]
        linked_lines = [candidate for candidate, candidate_range in matches.items() if candidate_range == matched_range]
        texts = tuple(document.lines[candidate].text for candidate in linked_lines)
        source_ids = tuple(dict.fromkeys(item.segment.id for item in group))
        similarity = text_similarity("\n".join(texts), "\n".join(item.transcription.text for item in group))
        if len(linked_lines) == 1:
            start, end, source, issues = _timing(group)
        else:
            position = linked_lines.index(index)
            start, end, issues = _interpolated_timing(texts, group, position)
            source = "interpolated"
        output.append(AlignedLyricLine(line.id, start, end, similarity, _score(texts, group), source, source_ids, issues))
    return LineAlignmentResult(tuple(output), "dp", bool(matches), (), ())
