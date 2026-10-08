import json
from pathlib import Path

from lazulite.alignment.line import align_lines
from lazulite.alignment.offset import OffsetAlignmentConfig, align_with_offset
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
    result = align_with_offset(document, transcript, analysis, config=OffsetAlignmentConfig(minimum_anchors=2))
    assert result.offset_accepted
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
    result = align_with_offset(document, transcript, analysis, config=OffsetAlignmentConfig(minimum_merged_section_chunks=2))

    assert result.offset_accepted
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


def test_whisper_probability_policy_distinguishes_low_tokens_and_missing_summary():
    from dataclasses import replace
    from lazulite.models.transcription import ProbabilitySummary,TranscriptionSegment
    from lazulite.transcription.evidence import transcription_quality,EvidenceConfig
    item=TranscriptionSegment('s','short lyric',None,None,None,None,None,(),{})
    uniform=replace(item,probability=ProbabilitySummary(.5,.5,0,10))
    low=replace(item,probability=ProbabilitySummary(.5,0,.5,10))
    config=EvidenceConfig('whisper-lexical-v1')
    assert transcription_quality(low,config=config).risk>transcription_quality(uniform,config=config).risk
    assert transcription_quality(item,config=config)==transcription_quality(item)
    assert transcription_quality(low)==transcription_quality(uniform)


def test_window_end_completion_caps_rest_and_preserves_repeated_chorus():
    from lazulite.models.transcription import TranscriptionSegment,TranscriptionResult,BackendInfo,RequestWindow,TimedText
    from lazulite.transcription.timing import organize_timing
    def item(i,start,end,core_end,onset):
        return TranscriptionSegment('s'+str(i),'repeat',None,None,None,None,None,
            (TimedText('repeat',onset,None,'word','model'),),{'raw_token_timestamps':[0,0]},
            RequestWindow(str(i),'s'+str(i),start,end,start,core_end))
    info=BackendInfo('test','m','r','v','f32')
    result=organize_timing(TranscriptionResult(info,(item(0,0,2,1.5,1),item(1,2,5,4,3))))
    assert result.segments[0].timed_text[0].end==1.5
    assert result.segments[0].timed_text[0].end_source=='acoustic_core'
    assert all('duplicate_padding' not in s.issues for s in result.segments)
    assert result.segments[0].backend_metadata['raw_token_timestamps']==[0,0]
    result=organize_timing(TranscriptionResult(info,(item(0,0,2,2,1.2),item(1,1,3,3,1.2))))
    assert 'duplicate_padding' in result.segments[1].issues


def test_line_coverage_is_not_a_reliability_promise():
    from lazulite.models.alignment import AlignedLyricLine,LineAlignmentResult
    result=LineAlignmentResult((AlignedLyricLine('a',1,None,.9,1,'model_units',('s',)),
        AlignedLyricLine('b',None,None,None,None,'unresolved',())), 'dp',False,())
    assert result.status=='partial'
    assert result.coverage=={'total':2,'matched':1,'complete':0,'single_endpoint':1,'unresolved':1}
    assert not hasattr(result,'reliable')


def test_mixed_real_units_and_coarse_blocks_keep_boundary_provenance():
    from lazulite.models.lyrics import LyricDocument,LyricLine
    from lazulite.models.audio import VocalSegment,VocalAnalysis
    from lazulite.models.transcription import BackendInfo,TranscriptionResult,TranscriptionSegment,TimedText
    from lazulite.alignment.line import align_lines
    info=BackendInfo('test','m','r','v','f32')
    a=VocalSegment('a',None,0,2,.5,1.5,None,None);b=VocalSegment('b',None,3,5,3.5,4.5,None,None)
    tr=TranscriptionResult(info,(
        TranscriptionSegment('a','first',None,None,None,None,None,(TimedText('first',.7,1.2,'word','model'),),{}),
        TranscriptionSegment('b','second',None,None,None,None,None,(),{})))
    result=align_lines(LyricDocument((LyricLine('a','first'),LyricLine('b','second'))),tr,VocalAnalysis('x',(a,b)))
    assert (result.lines[0].start,result.lines[0].end)==(.7,1.2)
    assert result.lines[1].start_source==result.lines[1].end_source=='acoustic_core'
