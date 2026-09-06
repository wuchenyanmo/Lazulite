import json
from pathlib import Path

from lazulite.alignment.line import align_lines
from lazulite.alignment.offset import align_with_offset
from lazulite.models.audio import VocalAnalysis, VocalSegment
from lazulite.models.lyrics import LyricDocument, LyricLine
from lazulite.models.transcription import BackendInfo, TranscriptionResult, TranscriptionSegment


def test_dp_alignment_keeps_unmatched_lines_and_uses_acoustic_core_without_model_time():
    analysis = VocalAnalysis("audio", (VocalSegment("s1", None, 1, 4, 1.5, 3.5, 1, 0),))
    transcript = TranscriptionResult(BackendInfo("fake", "fake", "r", "v", "float32"), (TranscriptionSegment("s1", "夜空を見上げて", "ja", None, None, None, None, (), {}),))
    result = align_lines(LyricDocument((LyricLine("L1", "夜空を見上げて"), LyricLine("L2", "存在しない"))), transcript, analysis)
    assert result.lines[0].timing_source == "acoustic_core"
    assert result.lines[0].start == 1.5
    assert result.lines[1].timing_source == "unresolved"


def test_offset_alignment_uses_core_start_without_missing_confidence_filter():
    analysis = VocalAnalysis("audio", (VocalSegment("s1", None, 11, 14, 11.5, 13.5, 1, 0), VocalSegment("s2", None, 21, 24, 21.5, 23.5, 1, 0)))
    document = LyricDocument((LyricLine("L1", "一", 1, 3), LyricLine("L2", "二", 11, 13)))
    transcript = TranscriptionResult(BackendInfo("fake", "fake", "r", "v", "float32"), (TranscriptionSegment("s1", "一", "ja", None, None, None, None, (), {}), TranscriptionSegment("s2", "二", "ja", None, None, None, None, (), {})))
    result = align_with_offset(document, transcript, analysis)
    assert result.reliable
    assert result.lines[0].start == 11.5


def test_dp_alignment_allows_bounded_multiple_lines_but_marks_coarse_time_as_interpolated():
    analysis = VocalAnalysis("audio", (VocalSegment("s1", None, 0, 4, 1, 3, 1, 0),))
    transcript = TranscriptionResult(
        BackendInfo("fake", "fake", "r", "v", "float32"),
        (TranscriptionSegment("s1", "一二", "ja", None, None, None, None, (), {}),),
    )
    result = align_lines(LyricDocument((LyricLine("L1", "一"), LyricLine("L2", "二"))), transcript, analysis)

    assert [line.timing_source for line in result.lines] == ["interpolated", "interpolated"]
    assert result.lines[0].start == 1
    assert result.lines[1].end == 3


def test_offset_alignment_uses_consistent_offsets_per_vocal_section():
    analysis = VocalAnalysis("audio", (
        VocalSegment("s1", "a", 10, 12, 10, 12, 1, 0),
        VocalSegment("s2", "a", 20, 22, 20, 22, 1, 0),
        VocalSegment("s3", "b", 50, 52, 50, 52, 1, 0),
        VocalSegment("s4", "b", 60, 62, 60, 62, 1, 0),
    ))
    document = LyricDocument((
        LyricLine("L1", "甲", 0), LyricLine("L2", "乙", 10),
        LyricLine("L3", "丙", 20), LyricLine("L4", "丁", 30),
    ))
    transcript = TranscriptionResult(
        BackendInfo("fake", "fake", "r", "v", "float32"),
        tuple(TranscriptionSegment(segment, text, "zh", None, None, None, None, (), {}) for segment, text in (("s1", "甲"), ("s2", "乙"), ("s3", "丙"), ("s4", "丁"))),
    )
    result = align_with_offset(document, transcript, analysis)

    assert result.reliable
    assert [line.start for line in result.lines] == [10, 20, 50, 60]
    assert {anchor.line_id for anchor in result.anchors} == {"L1", "L2", "L3", "L4"}


def test_frozen_generic_dp_regression_snapshot_keeps_every_line_traceable():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "alignment_regression.json").read_text(encoding="utf-8"))
    document = LyricDocument(tuple(LyricLine(**line) for line in fixture["document"]))
    analysis = VocalAnalysis("fixture", tuple(VocalSegment(**segment) for segment in fixture["segments"]))
    transcript = TranscriptionResult(
        BackendInfo(**fixture["backend"]),
        tuple(TranscriptionSegment(**segment) for segment in fixture["transcription"]),
    )

    result = align_lines(document, transcript, analysis)
    actual = [
        {"line_id": line.line_id, "start": line.start, "end": line.end, "timing_source": line.timing_source, "source_segment_ids": list(line.source_segment_ids)}
        for line in result.lines
    ]

    assert actual == fixture["expected"]
