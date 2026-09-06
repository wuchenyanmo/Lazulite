"""基于单调动态规划的行级歌词对齐。"""

from __future__ import annotations

from dataclasses import dataclass

from lazulite.lyrics.similarity import text_similarity
from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult, TranscriptionSegment


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """一次对齐使用的只读视图，不持有模型、网络或缓存。"""

    segment: VocalSegment
    transcription: TranscriptionSegment


@dataclass(frozen=True, slots=True)
class LineAlignmentConfig:
    minimum_similarity: float = 0.25
    skip_line_penalty: float = 0.65
    skip_evidence_penalty: float = 0.12


def _evidence(analysis: VocalAnalysis, transcription: TranscriptionResult) -> tuple[AlignmentEvidence, ...]:
    by_id = {segment.id: segment for segment in analysis.segments}
    return tuple(AlignmentEvidence(by_id[item.source_segment_id], item) for item in transcription.segments if item.source_segment_id in by_id and item.text.strip())


def _score(line: str, evidence: AlignmentEvidence) -> float:
    segment = evidence.segment
    value = 1.55 * text_similarity(line, evidence.transcription.text)
    if segment.presence is not None:
        value += 0.08 * segment.presence
    if segment.off_center_risk is not None:
        value -= 0.10 * segment.off_center_risk
    return value


def _timing(evidence: AlignmentEvidence) -> tuple[float | None, float | None, str, tuple[str, ...]]:
    units = evidence.transcription.timed_text
    starts = [unit.start for unit in units if unit.start is not None]
    ends = [unit.end for unit in units if unit.end is not None]
    issues: list[str] = []
    if starts:
        return min(starts), max(ends) if ends else None, "model_units", tuple(issues)
    issues.append("coarse_timing")
    if evidence.transcription.confidence is None:
        issues.append("missing_model_confidence")
    return evidence.segment.core_start, evidence.segment.core_end, "acoustic_core", tuple(issues)


def align_lines(document: LyricDocument, transcription: TranscriptionResult, analysis: VocalAnalysis, *, config: LineAlignmentConfig = LineAlignmentConfig()) -> LineAlignmentResult:
    """保留每个输入行；没有匹配证据的行显式标为 unresolved。"""
    evidence = _evidence(analysis, transcription)
    rows, columns = len(document.lines), len(evidence)
    dp = [[float("-inf")] * (columns + 1) for _ in range(rows + 1)]
    step = [[""] * (columns + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0
    for row in range(rows + 1):
        for column in range(columns + 1):
            current = dp[row][column]
            if current == float("-inf"):
                continue
            if row < rows and current - config.skip_line_penalty > dp[row + 1][column]:
                dp[row + 1][column] = current - config.skip_line_penalty; step[row + 1][column] = "line"
            if column < columns and current - config.skip_evidence_penalty > dp[row][column + 1]:
                dp[row][column + 1] = current - config.skip_evidence_penalty; step[row][column + 1] = "evidence"
            if row < rows and column < columns:
                similarity = text_similarity(document.lines[row].text, evidence[column].transcription.text)
                if similarity >= config.minimum_similarity and current + _score(document.lines[row].text, evidence[column]) > dp[row + 1][column + 1]:
                    dp[row + 1][column + 1] = current + _score(document.lines[row].text, evidence[column]); step[row + 1][column + 1] = "match"
    matched: dict[int, int] = {}; row, column = rows, columns
    while row or column:
        action = step[row][column]
        if action == "match": matched[row - 1] = column - 1; row -= 1; column -= 1
        elif action == "line": row -= 1
        elif action == "evidence": column -= 1
        else: break
    result: list[AlignedLyricLine] = []
    for index, line in enumerate(document.lines):
        if index not in matched:
            result.append(AlignedLyricLine(line.id, None, None, 0.0, None, "unresolved", (), ("low_text_match",)))
            continue
        item = evidence[matched[index]]; similarity = text_similarity(line.text, item.transcription.text)
        start, end, source, issues = _timing(item)
        result.append(AlignedLyricLine(line.id, start, end, similarity, _score(line.text, item), source, (item.segment.id,), issues))
    return LineAlignmentResult(tuple(result), "dp", bool(matched), (), ())
