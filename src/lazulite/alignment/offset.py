"""基于可追溯 anchor 的源时间偏移估计。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from lazulite.lyrics.similarity import text_similarity
from lazulite.models.alignment import AlignedLyricLine, AlignmentAnchor, LineAlignmentResult
from lazulite.models.audio import VocalAnalysis
from lazulite.models.lyrics import LyricDocument
from lazulite.models.transcription import TranscriptionResult


@dataclass(frozen=True, slots=True)
class OffsetAlignmentConfig:
    minimum_similarity: float = 0.62
    minimum_anchors: int = 2


def align_with_offset(document: LyricDocument, transcription: TranscriptionResult, analysis: VocalAnalysis, *, config: OffsetAlignmentConfig = OffsetAlignmentConfig()) -> LineAlignmentResult:
    """估计音频相对源歌词的偏移；锚点不足时不把原时间冒充自动成功。"""
    segments = {segment.id: segment for segment in analysis.segments}
    anchors: list[AlignmentAnchor] = []
    for item in transcription.segments:
        segment = segments.get(item.source_segment_id)
        if segment is None or not item.text.strip():
            continue
        candidates = [(text_similarity(line.text, item.text), line) for line in document.lines if line.source_start is not None]
        if not candidates:
            continue
        similarity, line = max(candidates, key=lambda pair: pair[0])
        if similarity < config.minimum_similarity:
            continue
        model_starts = [unit.start for unit in item.timed_text if unit.start is not None]
        observed = min(model_starts) if model_starts else segment.core_start
        anchors.append(AlignmentAnchor(line.id, segment.id, observed - float(line.source_start), similarity))
    reliable = len(anchors) >= config.minimum_anchors
    offset = median(anchor.offset for anchor in anchors) if reliable else None
    lines: list[AlignedLyricLine] = []
    for line in document.lines:
        if offset is None or line.source_start is None:
            lines.append(AlignedLyricLine(line.id, None, None, None, None, "unresolved", (), ("no_anchor",)))
            continue
        end = line.source_end + offset if line.source_end is not None else None
        lines.append(AlignedLyricLine(line.id, line.source_start + offset, end, None, None, "source_offset", tuple(anchor.source_segment_id for anchor in anchors if anchor.line_id == line.id), ()))
    return LineAlignmentResult(tuple(lines), "offset", reliable, tuple(anchors), () if reliable else ("insufficient_anchors",))
