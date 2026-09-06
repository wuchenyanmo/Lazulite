"""基于可追溯锚点的全局或分段源时间偏移估计。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from lazulite.lyrics.similarity import prefix_similarity, text_similarity
from lazulite.models.alignment import AlignedLyricLine, AlignmentAnchor, LineAlignmentResult
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult, TranscriptionSegment


@dataclass(frozen=True, slots=True)
class OffsetAlignmentConfig:
    minimum_similarity: float = 0.50
    minimum_prefix_similarity: float = 0.68
    minimum_anchors: int = 2
    minimum_section_anchors: int = 2
    maximum_delta_spread: float = 1.0
    maximum_section_mad: float = 0.35
    prefix_chars: int = 8


@dataclass(frozen=True, slots=True)
class _Candidate:
    line_index: int
    line_id: str
    segment: VocalSegment
    transcription: TranscriptionSegment
    similarity: float
    prefix: float
    offset: float

    @property
    def score(self) -> float:
        return 1.30 * self.prefix + self.similarity


def _observed_start(segment: VocalSegment, transcription: TranscriptionSegment) -> float:
    model_starts = [unit.start for unit in transcription.timed_text if unit.start is not None]
    return min(model_starts) if model_starts else segment.core_start


def _candidate_matrix(document: LyricDocument, transcription: TranscriptionResult, analysis: VocalAnalysis, config: OffsetAlignmentConfig, guidance: LineAlignmentResult | None) -> tuple[tuple[_Candidate | None, ...], ...]:
    segments = {segment.id: segment for segment in analysis.segments}
    timed_lines = [(index, line) for index, line in enumerate(document.lines) if line.source_start is not None]
    guided_sources = None if guidance is None else {line.line_id: set(line.source_segment_ids) for line in guidance.lines}
    rows: list[tuple[_Candidate | None, ...]] = []
    for item in transcription.segments:
        segment = segments.get(item.source_segment_id)
        if segment is None or not item.text.strip():
            continue
        candidates: list[_Candidate | None] = []
        observed = _observed_start(segment, item)
        for index, line in timed_lines:
            if guided_sources is not None and segment.id not in guided_sources.get(line.id, set()):
                candidates.append(None)
                continue
            full = text_similarity(line.text, item.text)
            prefix = prefix_similarity(line.text, item.text, prefix_chars=config.prefix_chars)
            if full < config.minimum_similarity or prefix < config.minimum_prefix_similarity:
                candidates.append(None)
                continue
            candidates.append(_Candidate(index, line.id, segment, item, full, prefix, observed - float(line.source_start)))
        rows.append(tuple(candidates))
    return tuple(rows)


def _select_monotonic(matrix: tuple[tuple[_Candidate | None, ...], ...]) -> tuple[_Candidate, ...]:
    """在候选矩阵中选一条单调最大得分路径，避免副歌锚点交叉。"""
    if not matrix:
        return ()
    rows, columns = len(matrix), len(matrix[0])
    scores = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    steps = [[""] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best, action = scores[row - 1][column], "evidence"
            if scores[row][column - 1] > best:
                best, action = scores[row][column - 1], "line"
            candidate = matrix[row - 1][column - 1]
            if candidate is not None and scores[row - 1][column - 1] + candidate.score > best:
                best, action = scores[row - 1][column - 1] + candidate.score, "match"
            scores[row][column], steps[row][column] = best, action
    selected: list[_Candidate] = []
    row, column = rows, columns
    while row and column:
        action = steps[row][column]
        if action == "match":
            candidate = matrix[row - 1][column - 1]
            if candidate is not None:
                selected.append(candidate)
            row -= 1; column -= 1
        elif action == "evidence":
            row -= 1
        else:
            column -= 1
    return tuple(reversed(selected))


def _largest_consistent_cluster(candidates: tuple[_Candidate, ...], maximum_spread: float) -> tuple[_Candidate, ...]:
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda item: item.offset)
    best: tuple[_Candidate, ...] = (ordered[0],)
    left = 0
    for right, item in enumerate(ordered):
        while item.offset - ordered[left].offset > maximum_spread:
            left += 1
        current = tuple(ordered[left:right + 1])
        if len(current) > len(best) or (len(current) == len(best) and sum(value.score for value in current) > sum(value.score for value in best)):
            best = current
    return best


def _estimate(candidates: tuple[_Candidate, ...], config: OffsetAlignmentConfig) -> tuple[float, tuple[_Candidate, ...]] | None:
    cluster = _largest_consistent_cluster(candidates, config.maximum_delta_spread)
    if len(cluster) < config.minimum_section_anchors:
        return None
    offset = float(median(item.offset for item in cluster))
    mad = float(median(abs(item.offset - offset) for item in cluster))
    if mad > config.maximum_section_mad:
        return None
    return offset, cluster


def _anchor(candidate: _Candidate) -> AlignmentAnchor:
    return AlignmentAnchor(candidate.line_id, candidate.segment.id, candidate.offset, candidate.score)


def align_with_offset(document: LyricDocument, transcription: TranscriptionResult, analysis: VocalAnalysis, *, config: OffsetAlignmentConfig = OffsetAlignmentConfig(), guidance: LineAlignmentResult | None = None) -> LineAlignmentResult:
    """估计偏移并按声学 section 分配；可用 DP 匹配限制重排锚点。"""
    source_lines = [(index, line) for index, line in enumerate(document.lines) if line.source_start is not None]
    if not source_lines:
        lines = tuple(AlignedLyricLine(line.id, None, None, None, None, "unresolved", (), ("missing_source_time",)) for line in document.lines)
        return LineAlignmentResult(lines, "offset", False, (), ("missing_source_timestamps",))
    matrix = _candidate_matrix(document, transcription, analysis, config, guidance)
    selected = _select_monotonic(matrix)
    by_section: dict[str | None, list[_Candidate]] = {}
    for candidate in selected:
        by_section.setdefault(candidate.segment.section_id, []).append(candidate)
    estimates: list[tuple[float, tuple[_Candidate, ...]]] = []
    for candidates in by_section.values():
        estimate = _estimate(tuple(candidates), config)
        if estimate is not None:
            estimates.append(estimate)
    reliable_candidates = tuple(item for _, candidates in estimates for item in candidates)
    if len(reliable_candidates) < config.minimum_anchors:
        global_estimate = _estimate(selected, config)
        estimates = [global_estimate] if global_estimate is not None else []
        reliable_candidates = () if global_estimate is None else global_estimate[1]
    if len(reliable_candidates) < config.minimum_anchors:
        lines = tuple(AlignedLyricLine(line.id, None, None, None, None, "unresolved", (), ("no_anchor",)) for line in document.lines)
        return LineAlignmentResult(lines, "offset", False, tuple(_anchor(item) for item in selected), ("insufficient_anchors",))

    # section 的歌词范围由可靠锚点决定；相邻 section 在中点处分界。
    ranges = sorted(
        ((min(item.line_index for item in candidates), max(item.line_index for item in candidates), offset, candidates) for offset, candidates in estimates),
        key=lambda item: item[0],
    )
    offsets: dict[int, tuple[float, tuple[_Candidate, ...]]] = {}
    boundaries = [((ranges[index][1] + ranges[index + 1][0]) // 2) for index in range(len(ranges) - 1)]
    start = 0
    for index, (_, _, offset, candidates) in enumerate(ranges):
        end = len(document.lines) - 1 if index == len(ranges) - 1 else boundaries[index]
        for line_index in range(start, end + 1):
            offsets[line_index] = (offset, candidates)
        start = end + 1

    anchors_by_line: dict[int, list[_Candidate]] = {}
    for candidate in reliable_candidates:
        anchors_by_line.setdefault(candidate.line_index, []).append(candidate)
    lines: list[AlignedLyricLine] = []
    for index, line in enumerate(document.lines):
        entry = offsets.get(index)
        if entry is None or line.source_start is None:
            lines.append(AlignedLyricLine(line.id, None, None, None, None, "unresolved", (), ("no_anchor",)))
            continue
        offset, candidates = entry
        direct = anchors_by_line.get(index, [])
        source_ids = tuple(dict.fromkeys(item.segment.id for item in direct))
        start = max(0.0, line.source_start + offset)
        end = None if line.source_end is None else max(start, line.source_end + offset)
        lines.append(AlignedLyricLine(
            line.id,
            start,
            end,
            max((item.similarity for item in direct), default=None),
            max((item.score for item in direct), default=None),
            "source_offset",
            source_ids,
            (),
        ))
    return LineAlignmentResult(lines=tuple(lines), strategy="offset", reliable=True, anchors=tuple(_anchor(item) for item in reliable_candidates), diagnostics=())
