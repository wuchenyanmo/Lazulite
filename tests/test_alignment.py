from lazulite.alignment.line import align_lines
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
