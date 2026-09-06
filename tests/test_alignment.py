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
