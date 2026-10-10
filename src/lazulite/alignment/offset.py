"""基于可追溯锚点的全局或分段源时间偏移估计。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from collections import Counter
from lazulite.lyrics.normalize import normalize_lyric_text
from lazulite.transcription.evidence import quality, transcription_quality, EvidenceConfig
from .line import align_lines, LineAlignmentConfig
from lazulite.transcription.timing import organize_timing

from lazulite.lyrics.similarity import prefix_similarity, text_similarity
from lazulite.models.alignment import (
    AlignedLyricLine,
    AlignmentAnchor,
    LineAlignmentResult,
)
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult, TranscriptionSegment


@dataclass(frozen=True, slots=True)
class OffsetAlignmentConfig:
    evidence: EvidenceConfig = EvidenceConfig()
    minimum_similarity: float = 0.50
    minimum_prefix_similarity: float = 0.68
    minimum_anchors: int = 3
    minimum_section_anchors: int = 2
    maximum_delta_spread: float = 1.0
    maximum_section_mad: float = 0.35
    prefix_chars: int = 8
    minimum_length_ratio: float = 0.55
    maximum_length_ratio: float = 1.8
    minimum_merged_section_chunks: int = 5
    maximum_section_merge_gap: float = 3.0

    def __post_init__(self):
        if (
            min(
                self.minimum_anchors,
                self.minimum_section_anchors,
                self.minimum_merged_section_chunks,
                self.prefix_chars,
            )
            < 1
        ):
            raise ValueError("anchor counts must be positive")
        if not 0 < self.minimum_length_ratio <= self.maximum_length_ratio:
            raise ValueError("invalid anchor length ratio")


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


def _observed_start(
    segment: VocalSegment, transcription: TranscriptionSegment
) -> float:
    model_starts = [
        unit.start for unit in transcription.timed_text if unit.start is not None
    ]
    return min(model_starts) if model_starts else segment.core_start


def _candidate_matrix(
    document: LyricDocument,
    transcription: TranscriptionResult,
    analysis: VocalAnalysis,
    config: OffsetAlignmentConfig,
    guidance: LineAlignmentResult | None,
) -> tuple[tuple[_Candidate | None, ...], ...]:
    segments = {segment.id: segment for segment in analysis.segments}
    timed_lines = [
        (index, line)
        for index, line in enumerate(document.lines)
        if line.source_start is not None
    ]
    guided_sources = (
        None
        if guidance is None
        else {line.line_id: set(line.source_segment_ids) for line in guidance.lines}
    )
    rows: list[tuple[_Candidate | None, ...]] = []
    for item in transcription.segments:
        segment = item.request_window or segments.get(item.source_segment_id)
        if (
            segment is None
            or not item.text.strip()
            or segment.core_start is None
            or "duplicate_padding" in item.issues
            or not transcription_quality(item, segment, config.evidence).usable
            or (segment.presence is not None and segment.presence < 0.28)
        ):
            continue
        candidates: list[_Candidate | None] = []
        observed = _observed_start(segment, item)
        for index, line in timed_lines:
            if guided_sources is not None and segment.id not in guided_sources.get(
                line.id, set()
            ):
                candidates.append(None)
                continue
            ratio = len(normalize_lyric_text(item.text)) / max(
                len(normalize_lyric_text(line.text)), 1
            )
            if not config.minimum_length_ratio <= ratio <= config.maximum_length_ratio:
                candidates.append(None)
                continue
            full = text_similarity(line.text, item.text)
            prefix = prefix_similarity(
                line.text, item.text, prefix_chars=config.prefix_chars
            )
            if (
                full < config.minimum_similarity
                or prefix < config.minimum_prefix_similarity
            ):
                candidates.append(None)
                continue
            candidates.append(
                _Candidate(
                    index,
                    line.id,
                    segment,
                    item,
                    full,
                    prefix,
                    observed - float(line.source_start),
                )
            )
        rows.append(tuple(candidates))
    return tuple(rows)


def _select_monotonic(
    matrix: tuple[tuple[_Candidate | None, ...], ...],
) -> tuple[_Candidate, ...]:
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
            if (
                candidate is not None
                and scores[row - 1][column - 1] + candidate.score > best
            ):
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
            row -= 1
            column -= 1
        elif action == "evidence":
            row -= 1
        else:
            column -= 1
    return tuple(reversed(selected))


def _largest_consistent_cluster(
    candidates: tuple[_Candidate, ...], maximum_spread: float
) -> tuple[_Candidate, ...]:
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda item: item.offset)
    best: tuple[_Candidate, ...] = (ordered[0],)
    left = 0
    for right, item in enumerate(ordered):
        while item.offset - ordered[left].offset > maximum_spread:
            left += 1
        current = tuple(ordered[left : right + 1])
        if len(current) > len(best) or (
            len(current) == len(best)
            and sum(value.score for value in current)
            > sum(value.score for value in best)
        ):
            best = current
    return best


def _estimate(
    candidates: tuple[_Candidate, ...], config: OffsetAlignmentConfig
) -> tuple[float, tuple[_Candidate, ...]] | None:
    cluster = _largest_consistent_cluster(candidates, config.maximum_delta_spread)
    if len(cluster) < config.minimum_section_anchors:
        return None
    offset = float(median(item.offset for item in cluster))
    mad = float(median(abs(item.offset - offset) for item in cluster))
    if mad > config.maximum_section_mad:
        return None
    return offset, cluster


def _anchor(candidate: _Candidate) -> AlignmentAnchor:
    return AlignmentAnchor(
        candidate.line_id, candidate.segment.id, candidate.offset, candidate.score,
        candidate.transcription.request_window.request_id if candidate.transcription.request_window is not None else None
    )


def merge_sections(analysis: VocalAnalysis, config: OffsetAlignmentConfig):
    """Return segment-to-group mapping, merging sparse consecutive sections."""
    sections = []
    for segment in sorted(analysis.segments, key=lambda item: item.start):
        if not sections or sections[-1][0] != segment.section_id:
            sections.append((segment.section_id, [segment]))
        else:
            sections[-1][1].append(segment)
    groups = []
    for _, members in sections:
        if not groups or (
            len(groups[-1]) >= config.minimum_merged_section_chunks
            and members[0].start - groups[-1][-1].end > config.maximum_section_merge_gap
        ):
            groups.append(list(members))
        else:
            groups[-1].extend(members)
    if len(groups) > 1 and len(groups[-1]) < config.minimum_merged_section_chunks:
        tail = groups.pop()
        groups[-1].extend(tail)
    return {
        segment.id: index for index, group in enumerate(groups) for segment in group
    }


def _guided_candidates(document, transcription, analysis, guidance, config):
    by_id = {s.id: s for s in analysis.segments}
    transcripts = {}
    for item in transcription.segments:
        transcripts.setdefault(item.source_segment_id, []).append(item)
    aligned = {line.line_id: line for line in guidance.lines}
    candidates = []
    for index, line in enumerate(document.lines):
        match = aligned.get(line.id)
        if (
            line.source_start is None
            or match is None
            or match.start is None
            or not match.source_segment_ids
        ):
            continue
        if match.timing_source == "interpolated":
            continue
        segment = by_id.get(match.source_segment_ids[0])
        items = [
            item
            for source in match.source_segment_ids
            for item in transcripts.get(source, ())
            if not match.source_request_ids or item.request_window is not None and item.request_window.request_id in match.source_request_ids
        ]
        if segment is None or not items:
            continue
        segment = items[0].request_window or segment
        text = match.matched_text or " ".join(item.text for item in items)
        if any(not transcription_quality(item,item.request_window or segment,config.evidence).usable for item in items) or (
            segment.presence is not None and segment.presence < 0.28
        ):
            continue
        ratio = len(normalize_lyric_text(text)) / max(
            len(normalize_lyric_text(line.text)), 1
        )
        full = text_similarity(line.text, text)
        prefix = prefix_similarity(line.text, text, prefix_chars=config.prefix_chars)
        if not config.minimum_length_ratio <= ratio <= config.maximum_length_ratio:
            continue
        if (
            full < config.minimum_similarity
            or prefix < config.minimum_prefix_similarity
        ):
            continue
        candidates.append(
            _Candidate(
                index,
                line.id,
                segment,
                items[0],
                full,
                prefix,
                match.start - line.source_start,
            )
        )
    return tuple(candidates)


def align_with_offset(
    document: LyricDocument,
    transcription: TranscriptionResult,
    analysis: VocalAnalysis,
    *,
    config: OffsetAlignmentConfig = OffsetAlignmentConfig(),
    guidance: LineAlignmentResult | None = None,
) -> LineAlignmentResult:
    """估计偏移并按声学 section 分配；可用 DP 匹配限制重排锚点。"""
    transcription=organize_timing(transcription,analysis)
    source_lines = [
        (index, line)
        for index, line in enumerate(document.lines)
        if line.source_start is not None
    ]
    if not source_lines:
        lines = tuple(
            AlignedLyricLine(
                line.id,
                None,
                None,
                None,
                None,
                "unresolved",
                (),
                ("missing_source_time",),
            )
            for line in document.lines
        )
        return LineAlignmentResult(
            lines, "offset", False, (), ("missing_source_timestamps",)
        )
    explicit_guidance = guidance is not None
    guidance = (
        guidance
        if guidance is not None
        else align_lines(document, transcription, analysis, config=LineAlignmentConfig(evidence=config.evidence))
    )
    matrix = _candidate_matrix(document, transcription, analysis, config, None)
    unguided = _select_monotonic(matrix)
    guided = _guided_candidates(document, transcription, analysis, guidance, config)
    selected = guided if guided or explicit_guidance else unguided
    section_by_segment = merge_sections(analysis, config)
    by_section: dict[int, list[_Candidate]] = {}
    for candidate in selected:
        by_section.setdefault(section_by_segment[candidate.segment.id], []).append(
            candidate
        )
    estimates: list[tuple[float, tuple[_Candidate, ...]]] = []
    for candidates in by_section.values():
        estimate = _estimate(tuple(candidates), config)
        if estimate is not None:
            estimates.append(estimate)
    reliable_candidates = tuple(
        item for _, candidates in estimates for item in candidates
    )
    if len(reliable_candidates) < config.minimum_anchors:
        global_estimate = _estimate(selected, config)
        estimates = [global_estimate] if global_estimate is not None else []
        reliable_candidates = () if global_estimate is None else global_estimate[1]
    if len(reliable_candidates) < config.minimum_anchors:
        lines = tuple(
            AlignedLyricLine(
                line.id, None, None, None, None, "unresolved", (), ("no_anchor",)
            )
            for line in document.lines
        )
        return LineAlignmentResult(
            lines,
            "offset",
            False,
            tuple(_anchor(item) for item in selected),
            ("insufficient_anchors",),
        )

    # Known DP section membership takes precedence over anchor-index midpoints.
    section_estimates = {}
    for entry in estimates:
        for candidate in entry[1]:
            section_estimates[section_by_segment[candidate.segment.id]] = entry
    aligned = {line.line_id: line for line in guidance.lines}
    offsets = {}
    unresolved_sections = set()
    known_sections = set()
    for index, line in enumerate(document.lines):
        match = aligned.get(line.id)
        groups = (
            [
                section_by_segment[sid]
                for sid in match.source_segment_ids
                if sid in section_by_segment
            ]
            if match
            else []
        )
        if groups:
            known_sections.add(index)
            section = Counter(groups).most_common(1)[0][0]
            if section in section_estimates:
                offsets[index] = section_estimates[section]
            else:
                unresolved_sections.add(index)
    # Only rows lacking section evidence may inherit the next/previous known section.
    next_entry = None
    for index in range(len(document.lines) - 1, -1, -1):
        if index in unresolved_sections:
            next_entry = None
        elif index in offsets:
            next_entry = offsets[index]
        elif next_entry is not None:
            offsets[index] = next_entry
    previous = None
    for index in range(len(document.lines)):
        if index in unresolved_sections:
            previous = None
        elif index in offsets:
            previous = offsets[index]
        elif previous is not None:
            offsets[index] = previous
    if not offsets and not unresolved_sections:
        ranges = sorted(
            estimates, key=lambda entry: min(c.line_index for c in entry[1])
        )
        boundaries = [
            (max(c.line_index for c in left[1]) + min(c.line_index for c in right[1]))
            // 2
            for left, right in zip(ranges, ranges[1:])
        ]
        first = 0
        for i, entry in enumerate(ranges):
            last = boundaries[i] if i < len(boundaries) else len(document.lines) - 1
            for index in range(first, last + 1):
                offsets[index] = entry
            first = last + 1

    anchors_by_line: dict[int, list[_Candidate]] = {}
    for candidate in reliable_candidates:
        anchors_by_line.setdefault(candidate.line_index, []).append(candidate)
    lines: list[AlignedLyricLine] = []
    for index, line in enumerate(document.lines):
        entry = offsets.get(index)
        if entry is None or line.source_start is None:
            lines.append(
                AlignedLyricLine(
                    line.id, None, None, None, None, "unresolved", (), ("no_anchor",)
                )
            )
            continue
        offset, candidates = entry
        direct = anchors_by_line.get(index, [])
        source_ids = tuple(dict.fromkeys(item.segment.id for item in direct))
        match = aligned.get(line.id)
        if not source_ids and match is not None:
            source_ids = match.source_segment_ids
        start = max(0.0, line.source_start + offset)
        end = None if line.source_end is None else max(start, line.source_end + offset)
        lines.append(
            AlignedLyricLine(
                line.id,
                start,
                end,
                max((item.similarity for item in direct), default=None),
                max((item.score for item in direct), default=None),
                "source_offset",
                source_ids,
                () if index in known_sections else ("inferred_section_offset",),
                match.matched_text if match is not None else None,
                "source_offset", "source_offset" if end is not None else None,
                match.source_request_ids if match is not None else (),
            )
        )
    complete = all(line.start is not None for line in lines)
    return LineAlignmentResult(
        lines=tuple(lines),
        strategy="offset",
        offset_accepted=complete,
        anchors=tuple(_anchor(item) for item in reliable_candidates),
        diagnostics=() if complete else ("unreliable_section",),
    )
