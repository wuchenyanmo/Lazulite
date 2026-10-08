"""基于单调动态规划的行级歌词对齐。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from lazulite.lyrics.similarity import text_similarity
from lazulite.lyrics.normalize import normalize_lyric_text
from lazulite.transcription.evidence import transcription_quality, EvidenceConfig, comparable_confidence
from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult, TranscriptionSegment, TimedText
from lazulite.transcription.timing import organize_timing


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """一次对齐使用的只读视图，不持有模型、网络或历史状态。"""

    segment: VocalSegment
    transcription: TranscriptionSegment


@dataclass(frozen=True, slots=True)
class LineAlignmentConfig:
    evidence: EvidenceConfig = EvidenceConfig()
    minimum_similarity: float = 0.08
    skip_line_penalty: float = 0.34
    skip_evidence_penalty: float = 0.24
    max_evidence_per_line: int = 2
    max_lines_per_evidence: int = 4

    def __post_init__(self) -> None:
        if self.max_evidence_per_line < 1 or self.max_lines_per_evidence < 1:
            raise ValueError("每次匹配的行数和分片数必须为正数")


def _evidence(
    analysis: VocalAnalysis, transcription: TranscriptionResult
) -> tuple[AlignmentEvidence, ...]:
    by_id = {segment.id: segment for segment in analysis.segments}
    return tuple(
        AlignmentEvidence(item.request_window or by_id[item.source_segment_id], item)
        for item in transcription.segments
        if item.source_segment_id in by_id and item.text.strip() and "duplicate_padding" not in item.issues
        and (item.request_window is None or item.request_window.core_start is not None)
    )


def _merge_penalty(line_count: int, evidence_count: int) -> float:
    return 0.05 * max(0, line_count - 1) + 0.04 * max(0, evidence_count - 1)


def _score(lines: tuple[str, ...], evidence: tuple[AlignmentEvidence, ...], policy=EvidenceConfig()) -> float:
    value = 1.55 * text_similarity(
        "\n".join(lines), "\n".join(item.transcription.text for item in evidence)
    )
    presence = [
        item.segment.presence for item in evidence if item.segment.presence is not None
    ]
    risk = [
        item.segment.off_center_risk
        for item in evidence
        if item.segment.off_center_risk is not None
    ]
    if presence:
        value += 0.08 * sum(presence) / len(presence)
    if risk:
        value -= 0.10 * sum(risk) / len(risk)
    confidences = [
        v
        for item in evidence
        if (v := comparable_confidence(item.transcription)) is not None
    ]
    if confidences:
        value += 0.16 * sum(confidences) / len(confidences)
    value -= (
        0.40
        * sum(transcription_quality(item.transcription, item.segment, policy).risk for item in evidence)
        / len(evidence)
    )
    return value - _merge_penalty(len(lines), len(evidence))


def _timing(
    evidence: tuple[AlignmentEvidence, ...],
) -> tuple[
    float | None, float | None, Literal["model_units", "acoustic_core"], tuple[str, ...]
]:
    units = tuple(unit for item in evidence for unit in item.transcription.timed_text)
    starts = [unit.start for unit in units if unit.start is not None]
    if starts:
        return min(starts), max((u.end for u in units if u.end is not None), default=None), "model_units", ()
    issues = ["coarse_timing"]
    if any(item.transcription.confidence is None for item in evidence):
        issues.append("missing_model_confidence")
    return (
        min(item.segment.core_start for item in evidence),
        max(item.segment.core_end for item in evidence),
        "acoustic_core",
        tuple(issues),
    )


def _interpolated_timing(
    lines: tuple[str, ...],
    evidence: tuple[AlignmentEvidence, ...],
    line_index: int,
) -> tuple[float, float, tuple[str, ...]]:
    """多行共用声学窗口时只做粗分配，明确不挪用 token 时间。"""
    start = min(item.segment.core_start for item in evidence)
    end = max(item.segment.core_end for item in evidence)
    weights = [max(len(normalize_lyric_text(line)), 1) for line in lines]
    total = sum(weights)
    leading = sum(weights[:line_index]) / total
    trailing = sum(weights[: line_index + 1]) / total
    return (
        start + (end - start) * leading,
        start + (end - start) * trailing,
        ("coarse_timing", "merged_line_evidence"),
    )


def _boundary_split(lines, evidence, config):
    """Search monotonic spans of real timed text, retaining trim/skip penalties."""
    if len(lines) < 2:
        return None
    entries=[]
    for item in evidence:
        valid=[u for u in item.transcription.timed_text if u.text.strip() and u.start is not None]
        if valid:
            entries.extend((u,item.segment.id) for u in valid)
        else:
            # A coarse block stays one segment-granularity item, never fabricated tokens.
            entries.append((TimedText(item.transcription.text,item.segment.core_start,item.segment.core_end,
                                      'segment','alignment','acoustic_core','acoustic_core'),item.segment.id))
    n, m = len(lines), len(entries)
    if m < n:
        return None
    span_text = {}
    span_similarity = {}

    def text(a, b):
        if (a, b) not in span_text:
            span_text[a, b] = "".join(unit.text for unit, _ in entries[a:b])
        return span_text[a, b]

    def similarity(i, a, b):
        key = i, a, b
        if key not in span_similarity:
            span_similarity[key] = text_similarity(lines[i], text(a, b))
        return span_similarity[key]

    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    back = {}
    for skip in range(m + 1):
        dp[0][skip] = -0.010 * skip
    for i in range(1, n + 1):
        prefix = []
        best = float("-inf")
        previous = None
        for pos in range(m + 1):
            value = dp[i - 1][pos] + 0.018 * pos
            if value > best:
                best, previous = value, pos
            prefix.append((best, previous))
        for b in range(i, m - (n - i) + 1):
            for a in range(i - 1, b):
                value, previous = prefix[a]
                if previous is None:
                    continue
                sim = similarity(i - 1, a, b)
                value += (
                    -0.018 * a
                    + 1.22 * sim
                    - (0.32 if sim < config.minimum_similarity else 0)
                )
                if value > dp[i][b]:
                    dp[i][b] = value
                    back[i, b] = previous, a
    last = max(range(n, m + 1), key=lambda b: dp[n][b] - 0.010 * (m - b))
    total = dp[n][last] - 0.010 * (m - last)
    spans = []
    for i in range(n, 0, -1):
        previous, a = back[i, last]
        chosen = entries[a:last]
        first_unit, last_unit = chosen[0][0], chosen[-1][0]
        if last_unit.end is not None and last_unit.end < first_unit.start:
            return None
        spans.append(
            (
                first_unit.start,
                last_unit.end,
                similarity(i - 1, a, last),
                tuple(dict.fromkeys(source for _, source in chosen)),
                text(a, last),
                first_unit.start_source or "model_timestamp",
                last_unit.end_source or ("model_timestamp" if last_unit.end is not None else None),
            )
        )
        last = previous
    spans.reverse()
    # Same non-text evidence terms as the coarse window branch.
    joined_similarity = text_similarity(
        "\n".join(lines), "\n".join(item.transcription.text for item in evidence)
    )
    total += _score(lines, evidence, config.evidence) - 1.55 * joined_similarity
    return total, spans


ActionKind = Literal["line", "evidence", "match"]
Action = tuple[ActionKind, int, int]


def align_lines(
    document: LyricDocument,
    transcription: TranscriptionResult,
    analysis: VocalAnalysis,
    *,
    config: LineAlignmentConfig = LineAlignmentConfig(),
) -> LineAlignmentResult:
    """保留所有输入行，并在受限范围内匹配多行与多分片。"""
    transcription = organize_timing(transcription, analysis)
    evidence = _evidence(analysis, transcription)
    rows, columns = len(document.lines), len(evidence)
    dp = [[float("-inf")] * (columns + 1) for _ in range(rows + 1)]
    step: list[list[Action | None]] = [[None] * (columns + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0
    candidates = {}

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
                update(
                    row + 1, column, current - config.skip_line_penalty, ("line", 1, 0)
                )
            if column < columns:
                update(
                    row,
                    column + 1,
                    current - config.skip_evidence_penalty,
                    ("evidence", 0, 1),
                )
            for line_count in range(
                1, min(config.max_lines_per_evidence, rows - row) + 1
            ):
                lines = tuple(
                    item.text for item in document.lines[row : row + line_count]
                )
                for evidence_count in range(
                    1, min(config.max_evidence_per_line, columns - column) + 1
                ):
                    group = evidence[column : column + evidence_count]
                    similarity = text_similarity(
                        "\n".join(lines),
                        "\n".join(item.transcription.text for item in group),
                    )
                    score = _score(lines, group, config.evidence) - (
                        0.32 if similarity < config.minimum_similarity else 0
                    )
                    split = None
                    gate = 0.18 if line_count == 2 else 0.24
                    if (
                        line_count >= 2
                        and (line_count == 2 or evidence_count == 1)
                        and similarity >= gate
                    ):
                        trial = _boundary_split(lines, group, config)
                        if trial is not None and trial[0] > score:
                            score, split = trial
                    candidates[row, column, line_count, evidence_count] = score, split
                    update(
                        row + line_count,
                        column + evidence_count,
                        current + score,
                        ("match", line_count, evidence_count),
                    )

    matches: dict[int, tuple[int, int]] = {}
    assignments = {}
    row, column = rows, columns
    while row or column:
        action = step[row][column]
        if action is None:
            break
        kind, line_count, evidence_count = action
        if kind == "match":
            score, split = candidates[
                row - line_count, column - evidence_count, line_count, evidence_count
            ]
            for position, index in enumerate(range(row - line_count, row)):
                matches[index] = (column - evidence_count, column)
                assignments[index] = (score, split[position] if split else None)
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
            output.append(
                AlignedLyricLine(
                    line.id,
                    None,
                    None,
                    0.0,
                    None,
                    "unresolved",
                    (),
                    ("low_text_match",),
                )
            )
            continue
        group = evidence[matched_range[0] : matched_range[1]]
        linked_lines = [
            candidate
            for candidate, candidate_range in matches.items()
            if candidate_range == matched_range
        ]
        texts = tuple(document.lines[candidate].text for candidate in linked_lines)
        source_ids = tuple(dict.fromkeys(item.segment.id for item in group))
        similarity = text_similarity(
            "\n".join(texts), "\n".join(item.transcription.text for item in group)
        )
        score, split = assignments[index]
        matched_text = " ".join(item.transcription.text for item in group)
        start_source=end_source=None
        if split is not None:
            start, end, similarity, source_ids, matched_text, start_source, end_source = split
            source, issues = ("acoustic_core", ("coarse_timing",)) if start_source==end_source=="acoustic_core" else ("model_units", ())
        elif len(linked_lines) == 1:
            start, end, source, issues = _timing(group)
        else:
            position = linked_lines.index(index)
            start, end, issues = _interpolated_timing(texts, group, position)
            source = "interpolated"
        if start_source is None and start is not None:
            starts=[u for item in group for u in item.transcription.timed_text if u.start==start]
            start_source=(starts[0].start_source or 'model_timestamp') if starts and source=='model_units' else source
        if end_source is None and end is not None:
            ends=[u for item in group for u in item.transcription.timed_text if u.end==end]
            end_source=(ends[-1].end_source or 'model_timestamp') if ends and source=='model_units' else source
        issues += tuple(dict.fromkeys(issue for item in group for issue in item.transcription.issues))
        if similarity < config.minimum_similarity:
            issues += ("low_text_match",)
        if any(
            not transcription_quality(item.transcription, item.segment, config.evidence).usable for item in group
        ):
            issues += ("heuristic_quality_risk",)
        output.append(
            AlignedLyricLine(
                line.id,
                start,
                end,
                similarity,
                score,
                source,
                source_ids,
                issues,
                matched_text,
                start_source,end_source,
                tuple(item.transcription.request_window.request_id for item in group if item.transcription.request_window is not None),
            )
        )
    return LineAlignmentResult(tuple(output), "dp", False, (), ())
