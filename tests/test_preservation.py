"""Algorithm-loss counterexamples; expectations are behavioral, not implementation snapshots."""

from dataclasses import replace
from pathlib import Path
from threading import Barrier
import numpy as np
import pytest
from lazulite.audio.activity import VocalConfig, split_long_interval, vocal_evidence
from lazulite.audio.io import resample_audio
from lazulite.alignment.line import align_lines
from lazulite.alignment.offset import OffsetAlignmentConfig, align_with_offset
from lazulite.lyrics.lrc import (
    read_lrc,
    read_plain_text,
    with_translation,
    write_lrc,
    clean_translation,
)
from lazulite.lyrics.providers.base import SearchCandidate, FetchedLyricCandidate
from lazulite.lyrics.providers.qqmusic import QQMusicProvider
from lazulite.lyrics.service import rank_lyrics, search_lyrics, _content_score
from lazulite.models.audio import (
    AudioBuffer,
    AudioMetadata,
    VocalSegment,
    VocalAnalysis,
)
from lazulite.models.lyrics import LyricDocument, LyricLine
from lazulite.models.transcription import (
    BackendInfo,
    TranscriptionResult,
    TranscriptionSegment,
    TimedText,
)
from lazulite.pipeline import AlignmentPipeline, PipelineConfig
from lazulite.transcription.base import TranscriberCapabilities, TranscriptionRequest
from lazulite.transcription.evidence import quality, comparable_confidence
from lazulite.transcription.run import TranscriptionConfig, transcribe_segments

INFO = BackendInfo("fixture", "fixture", "v1", "offline", "float32")


def segment(i, start=0, end=10, section="A", presence=1):
    return VocalSegment(str(i), section, start, end, start, end, presence, 0)


def transcript(i, text, units=()):
    return TranscriptionSegment(str(i), text, None, None, None, None, None, units, {})


@pytest.mark.parametrize("frequency,expected", [(1000, 0.7071), (12000, 0)])
def test_resampling_rejects_alias_and_retains_passband(frequency, expected):
    wave = np.sin(2 * np.pi * frequency * np.arange(48000) / 48000).astype(np.float32)
    out = resample_audio(
        AudioBuffer(np.stack([wave, -wave]), 48000, "tone"), sample_rate=16000
    )
    assert out.samples.shape == (2, 16000)
    assert np.sqrt(np.mean(out.samples[0, 800:-800] ** 2)) == pytest.approx(
        expected, abs=0.002
    )
    np.testing.assert_allclose(out.samples[0], -out.samples[1], atol=1e-6)


def test_resampling_preserves_impulse_origin_and_length():
    wave = np.zeros((1, 48000), np.float32)
    wave[0, 24000] = 1
    out = resample_audio(AudioBuffer(wave, 48000, "impulse"), sample_rate=16000)
    assert np.argmax(out.samples) == 8000
    assert out.samples.shape == (1, 16000)


@pytest.mark.parametrize("with_valley", [False, True])
def test_split_has_contiguous_short_tail_and_uses_valley(with_valley):
    config = VocalConfig(
        hop_length=1, frame_length=2, min_segment_seconds=0.45, max_segment_seconds=8
    )
    activity = np.ones(820, np.float32)
    if with_valley:
        activity[380:421] = 0
    pieces = split_long_interval(config, 0, len(activity), activity, activity, 100)
    assert pieces[0][0] == 0 and pieces[-1][1] == 820
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))
    assert all(0 < b - a <= 800 for a, b in pieces)
    if with_valley:
        assert 380 <= pieces[0][1] <= 421


@pytest.mark.parametrize(
    "seconds,amplitude,expected", [(4, 0.2, False), (8, 0.2, True), (8, 0.001, False)]
)
def test_track_gate_independent_of_local_segments(seconds, amplitude, expected):
    audio = AudioBuffer(
        np.full((1, 16000 * seconds), amplitude, np.float32), 16000, "gate"
    )
    assert vocal_evidence(audio)[0] is expected


def test_reference_and_relative_duration_gate():
    wave = np.zeros((1, 40 * 16000), np.float32)
    wave[:, 2 * 16000 : 8 * 16000] = 0.02
    audio = AudioBuffer(wave, 16000, "stem")
    reference = AudioBuffer(np.full_like(wave, 0.1), 16000, "original")
    assert (
        vocal_evidence(audio, reference=reference)[0] is False
    )  # six seconds < 20 percent
    assert (
        vocal_evidence(
            audio, reference=reference, config=VocalConfig(min_vocal_ratio=0.1)
        )[0]
        is True
    )


class Backend:
    info = INFO
    capabilities = TranscriberCapabilities(
        "none", context=False, max_audio_seconds=None
    )

    def __init__(self):
        self.calls = []

    def transcribe(self, inputs, *, request):
        self.calls.append((tuple(inputs), request))
        return tuple(transcript(i.segment.id, "hello world") for i in inputs)


def test_short_non_silent_track_skips_asr_and_keeps_lyrics():
    audio = AudioBuffer(np.full((1, 3 * 16000), 0.3, np.float32), 16000, "short")
    backend = Backend()
    result = AlignmentPipeline(transcriber=backend).run(audio, read_plain_text("hello"))
    assert not result.analysis.is_vocal and not backend.calls
    assert result.alignment.lines[0].start is None


def test_actual_asr_input_is_main_vocal(monkeypatch):
    import lazulite.pipeline as pipeline

    wave = np.stack(
        [np.linspace(-1, 1, 128000), np.linspace(0.8, -0.8, 128000)]
    ).astype(np.float32)
    audio = AudioBuffer(wave, 16000, "stereo")
    monkeypatch.setattr(
        __import__("lazulite.audio.preparation", fromlist=["analyze_vocals"]),
        "analyze_vocals",
        lambda *a, **k: VocalAnalysis("stereo", (segment(0, end=8),), (), True, 8),
    )
    backend = Backend()
    AlignmentPipeline(transcriber=backend).run(
        audio, read_plain_text("hello"), config=PipelineConfig(mode="dp")
    )
    expected = (wave[0] + wave[1]) / 2 - np.abs((wave[0] - wave[1]) / 2)
    np.testing.assert_allclose(backend.calls[0][0][0].audio.samples[0], expected)


def test_nonuniform_multiline_uses_real_units_and_local_acoustic_end():
    units = (
        TimedText("aaaa", 1, 2, "word", "model"),
        TimedText("bbbb", 7, None, "word", "model"),
    )
    result = align_lines(
        read_plain_text("aaaa\nbbbb"),
        TranscriptionResult(INFO, (transcript(0, "aaaa bbbb", units),)),
        VocalAnalysis("x", (segment(0),)),
    )
    assert [(l.start, l.end, l.timing_source) for l in result.lines] == [
        (1, 2, "model_units"),
        (7, 10, "model_units"),
    ]
    assert [l.matched_text for l in result.lines] == ["aaaa", "bbbb"]


def test_default_two_anchors_are_insufficient_and_zero_offset_is_valid():
    texts = ["aaaa", "bbbb", "cccc"]
    doc = LyricDocument(
        tuple(LyricLine(str(i), t, i * 10) for i, t in enumerate(texts))
    )
    trans = TranscriptionResult(
        INFO, tuple(transcript(i, t) for i, t in enumerate(texts))
    )
    analysis = VocalAnalysis(
        "x", tuple(segment(i, i * 10, i * 10 + 2) for i in range(3))
    )
    assert not align_with_offset(
        replace(doc, lines=doc.lines[:2]),
        replace(trans, segments=trans.segments[:2]),
        replace(analysis, segments=analysis.segments[:2]),
    ).offset_accepted
    result = align_with_offset(doc, trans, analysis)
    assert result.offset_accepted and all(a.offset == 0 for a in result.anchors)


def test_quality_preserves_unknown_confidence_and_rejects_repetition():
    item = transcript(0, "hello world")
    assert comparable_confidence(item) is None
    assert (
        comparable_confidence(
            replace(
                item, confidence=0, confidence_kind="generation_token_geometric_mean"
            )
        )
        == 0
    )
    assert quality(item.text).usable
    assert not quality("hello world hello world").usable


def test_task_context_is_local_and_timing_disables_prompt():
    class ContextBackend(Backend):
        capabilities = TranscriberCapabilities(
            "word", context=True, max_audio_seconds=None
        )

    backend = ContextBackend()
    audio = AudioBuffer(np.zeros((1, 32000), np.float32), 16000, "a")
    segments = (segment(0, 0, 1), segment(1, 1, 2))
    for _ in range(2):
        transcribe_segments(audio, segments, transcriber=backend)
    assert [r.context for _, r in backend.calls] == [
        None,
        "hello world",
        None,
        "hello world",
    ]
    transcribe_segments(
        audio,
        segments,
        transcriber=backend,
        request=TranscriptionRequest(include_timing=True),
    )
    assert backend.calls[-1][1].context is None


def test_auto_fallback_only_requests_missing_stateless_windows(monkeypatch):
    import lazulite.pipeline as pipeline

    segments = tuple(segment(i, i * 2, i * 2 + 2) for i in range(5))
    monkeypatch.setattr(
        __import__("lazulite.audio.preparation", fromlist=["analyze_vocals"]),
        "analyze_vocals",
        lambda *a, **k: VocalAnalysis("a", segments, (), True, 10),
    )
    backend = Backend()
    audio = AudioBuffer(np.zeros((1, 160000), np.float32), 16000, "a")
    doc = LyricDocument(
        tuple(LyricLine(str(i), "unrelated lyric", i * 2) for i in range(5))
    )
    result = AlignmentPipeline(transcriber=backend).run(audio, doc)
    assert [[i.segment.id for i in inputs] for inputs, _ in backend.calls] == [
        ["0", "1", "2"],
        ["3", "4"],
    ]
    assert "offset_fallback" in result.alignment.diagnostics
    assert len(result.execution["requests"]) == 2


def test_parser_preserves_roles_prose_and_translation():
    text = (
        "[00:10.00]A: "
        + 35 * "a"
        + "\n[00:20.00]你：你好\n[00:30.00]Instrumental\n[00:40.00]My instinct"
    )
    doc = read_lrc(text)
    assert [(l.singer, l.text) for l in doc.document.lines] == [
        ("A", "a" * 35),
        (None, "你:你好"),
        (None, "My instinct"),
    ]
    assert read_plain_text("男: hello").lines[0].singer == "男"
    assert read_lrc(write_lrc(doc)).document == doc.document
    translated = with_translation(doc, "[00:10.00]你：你好\n[00:20.00]【世界】")
    assert translated.translations[0][1] == "你：你好"
    assert translated.translations[1][1] == "【世界】"
    bracketed = replace(
        doc, translations=(("line-0", "【你好】"), ("line-1", "【世界】"))
    )
    assert clean_translation(bracketed).translations == (
        ("line-0", "你好"),
        ("line-1", "世界"),
    )
    assert bracketed.translations[0][1] == "【你好】"


def test_content_traditional_score_and_qualified_timestamp_order():
    def fetched(key, score, text):
        return FetchedLyricCandidate(
            SearchCandidate("fixture", key, key, None, None, 1, score),
            read_lrc(text),
            1,
        )

    trans = TranscriptionResult(INFO, (transcript(0, "爱与梦"),))
    high = fetched("high", 100, "[00:01.00]愛與夢")
    low = fetched("low", 0, "[00:01.00]xxx")
    assert _content_score(high, trans) == pytest.approx(1)
    assert [
        r.item.candidate.candidate_id
        for r in rank_lyrics((low, high), trans, providers=())
    ] == ["high"]
    untimed = replace(
        high,
        candidate=replace(high.candidate, candidate_id="untimed"),
        lyrics=replace(high.lyrics, document=read_plain_text("愛與夢")),
    )
    timed = replace(
        high, candidate=replace(high.candidate, candidate_id="timed", match_score=80)
    )
    assert (
        rank_lyrics((untimed, timed), trans, providers=())[
            0
        ].item.candidate.candidate_id
        == "timed"
    )


def test_provider_search_runs_concurrently_with_stable_order():
    barrier = Barrier(2)

    class Provider:
        priority = 1

        def __init__(self, name):
            self.source_name = name

        def search(self, **kwargs):
            barrier.wait(timeout=3)
            return [
                SearchCandidate(self.source_name, "1", "title", None, None, 10, 100)
            ]

    result = search_lyrics(
        AudioMetadata(Path("x"), "title", None, None, 10),
        providers=(Provider("b"), Provider("a")),
    )
    assert [c.source for c in result] == ["a", "b"]


def test_qq_primary_fallback_and_failure():
    import base64

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class Session:
        def __init__(self, primary, fallback):
            self.primary = primary
            self.fallback = fallback
            self.calls = []

        def post(self, *a, **k):
            self.calls.append("post")
            return Response(self.primary)

        def get(self, *a, **k):
            self.calls.append("get")
            return Response(self.fallback)

    candidate = SearchCandidate(
        "qqmusic", "mid", "title", None, None, 10, 100, {"id": 1}
    )
    payload = {
        "code": 0,
        "req_0": {
            "code": 0,
            "data": {
                "crypt": 0,
                "qrc": 0,
                "lyric": base64.b64encode(b"[00:10.00]hello").decode(),
            },
        },
    }
    session = Session(payload, {"retcode": -1})
    assert (
        QQMusicProvider(session=session).fetch(candidate).document.lines[0].text
        == "hello"
    )
    assert session.calls == ["post"]
    session = Session({"code": -1}, {"retcode": 0, "lyric": "[00:10.00]fallback"})
    assert (
        QQMusicProvider(session=session).fetch(candidate).document.lines[0].text
        == "fallback"
    )
    assert session.calls == ["post", "get"]
    with pytest.raises(RuntimeError):
        QQMusicProvider(session=Session({"code": -1}, {"retcode": -1})).fetch(candidate)


def test_dp_section_membership_overrides_anchor_midpoints():
    from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult

    texts = ["aaaa", "bbbb", "eeee", "ffff", "cccc", "dddd"]
    sources = [10, 20, 30, 40, 100, 110]
    starts = [12, 22, 32, 42, 120, 130]
    doc = LyricDocument(
        tuple(
            LyricLine(str(i), text, start)
            for i, (text, start) in enumerate(zip(texts, sources))
        )
    )
    analysis = VocalAnalysis(
        "x",
        tuple(
            segment(
                i, start, start + 2, "A" if i < 4 else "B", 0.1 if i in (2, 3) else 1
            )
            for i, start in enumerate(starts)
        ),
    )
    trans = TranscriptionResult(
        INFO, tuple(transcript(i, text) for i, text in enumerate(texts))
    )
    guidance = LineAlignmentResult(
        tuple(
            AlignedLyricLine(
                str(i), start, start + 2, 1, 1, "acoustic_core", (str(i),), (), text
            )
            for i, (text, start) in enumerate(zip(texts, starts))
        ),
        "dp",
        True,
        (),
    )
    result = align_with_offset(
        doc,
        trans,
        analysis,
        guidance=guidance,
        config=OffsetAlignmentConfig(minimum_merged_section_chunks=2),
    )
    assert result.offset_accepted
    assert [line.start - source for line, source in zip(result.lines, sources)] == [
        2,
        2,
        2,
        2,
        20,
        20,
    ]


def test_guided_multiline_anchors_use_each_lines_onset():
    doc = LyricDocument(
        tuple(
            LyricLine(str(i), text, start)
            for i, (text, start) in enumerate(zip(["aaaa", "bbbb", "cccc"], [0, 3, 6]))
        )
    )
    units = tuple(
        TimedText(text, start, start + 1, "word", "model")
        for text, start in zip(["aaaa", "bbbb", "cccc"], [2, 5, 8])
    )
    trans = TranscriptionResult(INFO, (transcript(0, "aaaa bbbb cccc", units),))
    result = align_with_offset(doc, trans, VocalAnalysis("x", (segment(0),)))
    assert result.offset_accepted
    assert [line.start for line in result.lines] == [2, 5, 8]
    assert [a.offset for a in result.anchors] == [2, 2, 2]


def test_qq_search_requires_detail_duration_before_scoring():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            pass

    class Session:
        def get(self, *a, **k):
            return Response(
                {"code": 0, "data": {"song": {"itemlist": [{"id": 123, "mid": "mid"}]}}}
            )

        def post(self, *a, **k):
            assert k["json"]["req_0"]["param"] == {"song_id": 123}
            return Response(
                {
                    "code": 0,
                    "req_0": {
                        "code": 0,
                        "data": {
                            "track_info": {
                                "id": 123,
                                "mid": "mid",
                                "name": "song",
                                "interval": 60,
                                "singer": [{"name": "artist"}],
                                "album": {"name": "different", "subtitle": "album"},
                            }
                        },
                    },
                }
            )

    candidates = QQMusicProvider(session=Session()).search(
        "song", 60, "artist", "album"
    )
    assert candidates[0].duration == 60 and candidates[0].match_score == 100


def test_old_cache_schema_misses_and_context_config_changes_key(tmp_path):
    from lazulite.cache import FileCache

    class CachedBackend(Backend):
        cache_identity = {"fixture": "v1"}

    audio = AudioBuffer(np.zeros((1, 16000), np.float32), 16000, "sha256:a")
    pipeline = AlignmentPipeline(transcriber=CachedBackend(), cache=FileCache(tmp_path))
    doc = read_plain_text("hello")
    assert pipeline._deserialize({"schema": "pipeline-result-v1", "result": {}}) is None
    assert pipeline._cache_key(audio, doc, PipelineConfig()) != pipeline._cache_key(
        audio,
        doc,
        PipelineConfig(transcription=TranscriptionConfig(prompt_mode="none")),
    )


@pytest.mark.parametrize("length", [1, 2, 10, 1001])
def test_resampling_short_clips_are_not_lost(length):
    import math

    audio = AudioBuffer(np.ones((1, length), np.float32), 44100, "short")
    result = resample_audio(audio, sample_rate=16000)
    assert result.samples.shape == (1, math.ceil(length * 16000 / 44100))


def test_default_providers_do_not_report_network_failure_as_empty_success():
    from requests.exceptions import ConnectionError
    from lazulite.lyrics.providers import NeteaseProvider, KugouProvider, LRCLIBProvider

    class Broken:
        def get(self, *args, **kwargs):
            raise ConnectionError("offline fixture")

    providers = tuple(
        cls(session=Broken())
        for cls in (NeteaseProvider, KugouProvider, LRCLIBProvider, QQMusicProvider)
    )
    with pytest.raises(RuntimeError, match="所有歌词来源搜索失败"):
        search_lyrics(
            AudioMetadata(Path("x"), "title", None, None, 10), providers=providers
        )


def test_cli_passes_actual_stereo_44100_to_separator(monkeypatch, tmp_path):
    import wave
    from types import SimpleNamespace
    import lazulite.cli as cli
    import lazulite.audio.separation as separation
    import lazulite.lyrics.ttml as ttml

    source = tmp_path / "stereo.wav"
    values = np.stack([np.full(8000, 1000), np.full(8000, -2000)], axis=1).astype("<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(values.tobytes())
    received = []

    class Separator:
        def __init__(self, config):
            pass

        def separate(self, audio):
            received.append(audio)
            return audio

        def close(self):
            pass

    backend = Backend()
    backend.close = lambda: None
    monkeypatch.setattr(cli, "_build_transcriber", lambda args: backend)
    monkeypatch.setattr(separation, "DemucsSeparator", Separator)
    monkeypatch.setattr(
        ttml, "read_ttml_for_alignment", lambda path: read_plain_text("hello")
    )
    cli._run_align(
        SimpleNamespace(
            audio=source,
            lyrics=tmp_path / "x.ttml",
            output=tmp_path / "result.json",
            without_separation=False,
            separator_model="fixture",
            separator_device="cpu",
            mode="dp",
            language=None,
        )
    )
    assert received[0].sample_rate == 44100
    assert received[0].samples.shape == (2, 44100)
    assert not np.array_equal(received[0].samples[0], received[0].samples[1])


def test_cache_round_trip_preserves_line_evidence_and_execution():
    import json
    from lazulite.models.alignment import AlignedLyricLine, LineAlignmentResult
    from lazulite.pipeline import PipelineResult

    result = PipelineResult(
        VocalAnalysis("x", (), (), True, 8),
        None,
        LineAlignmentResult(
            (
                AlignedLyricLine(
                    "1",
                    1,
                    None,
                    1,
                    1,
                    "model_units",
                    ("v1",),
                    ("missing_end",),
                    "hello",
                ),
            ),
            "dp",
            True,
            (),
        ),
        {"requests": [{"segments": ["v1"], "include_timing": True}]},
    )
    payload = json.loads(json.dumps(AlignmentPipeline._serialize(result)))
    assert AlignmentPipeline._deserialize(payload) == result


def test_qq_missing_duration_does_not_become_zero_scored_success(monkeypatch):
    provider = QQMusicProvider()
    monkeypatch.setattr(
        provider,
        "_request",
        lambda *a, **k: {
            "code": 0,
            "data": {"song": {"itemlist": [{"id": 1, "mid": "mid"}]}},
        },
    )
    monkeypatch.setattr(
        provider,
        "_musicu",
        lambda *a, **k: {"track_info": {"mid": "mid", "name": "song"}},
    )
    with pytest.raises(RuntimeError, match="详情全部获取失败"):
        provider.search("song", 60)
