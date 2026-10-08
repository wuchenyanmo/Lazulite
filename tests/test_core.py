"""不依赖模型栈的核心类型回归测试。"""

from __future__ import annotations

import importlib
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.lyrics import LyricDocument, LyricLine, ReadingSpan
from lazulite.models.transcription import TranscriptionSegment
from lazulite.lyrics.lrc import read_lrc, with_translation, write_lrc
from lazulite.lyrics.normalize import normalize_lyric_text
from lazulite.lyrics.ttml import TtmlFormatError, read_ttml_for_alignment
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate
from lazulite.lyrics.similarity import text_similarity
from lazulite.lyrics.providers.netease import (
    candidate_from_netease_result,
    lyrics_from_netease_payload,
)
from lazulite.audio.io import decode_audio, read_audio_metadata, resample_audio
from lazulite.audio.activity import VocalConfig, analyze_vocals
from lazulite.lyrics.providers.qqmusic import candidate_from_qq_result, lyrics_from_qq_payload
from lazulite.lyrics.providers.lrclib import lyrics_from_lrclib_payload
from lazulite.lyrics.providers.base import SearchCandidate
from lazulite.lyrics.service import fetch_lyrics, rank_lyrics, search_lyrics
from lazulite.models.transcription import BackendInfo, TranscriptionResult


class _FakeProvider:
    source_name = "fake"
    priority = 5

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def search(self, title, duration, artist=None, album=None, **kwargs):
        self.calls.append((title, artist, album))
        return [
            SearchCandidate("fake", "same", "夜奏花", "YURiKA", None, duration, 80.0),
            SearchCandidate("fake", "same", "夜奏花", "YURiKA", None, duration, 90.0),
        ]

    def fetch(self, candidate):
        return read_lrc("[00:01.00]夜空を見上げて")


def test_base_package_import_does_not_load_model_stacks() -> None:
    sys.modules.pop("lazulite", None)
    importlib.import_module("lazulite")

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_audio_and_vocal_segment_keep_explicit_layout_and_windows() -> None:
    audio = AudioBuffer(
        samples=np.zeros((2, 48_000), dtype=np.float32),
        sample_rate=48_000,
        source_id="sha256:example",
    )
    segment = VocalSegment("v1", "verse-1", 10.0, 14.0, 10.5, 13.5, 0.8, 0.1)

    assert audio.duration == 1.0
    assert segment.core_start == 10.5
    with pytest.raises(ValueError, match="float32"):
        AudioBuffer(np.zeros((1, 4), dtype=np.float64), 48_000, "source")


def test_lyrics_preserve_original_text_and_code_point_ranges() -> None:
    document = LyricDocument((LyricLine("L4", "今日も歌う"),), language="ja")
    span = ReadingSpan("L4", 0, 2, "きょう")

    assert document.lines[0].text == "今日も歌う"
    assert span.base_end == 2
    with pytest.raises(ValueError, match="唯一"):
        LyricDocument((LyricLine("L4", "一"), LyricLine("L4", "二")))


def test_missing_confidence_is_distinct_from_zero() -> None:
    missing = TranscriptionSegment("v1", "歌", "ja", None, None, None, None, (), {})
    zero = TranscriptionSegment("v2", "歌", "ja", None, None, 0.0, "external", (), {})

    assert missing.confidence is None
    assert zero.confidence == 0.0


def test_lrc_parses_metadata_singer_and_translation_without_rewriting_original_text() -> None:
    lrc = read_lrc("[ti:示例]\n[00:01.00]作词：作者\n[00:02.00]【合】今日(きょう)、歌う")
    translated = with_translation(lrc, "[00:02.04]Today, we sing")

    line = translated.document.lines[0]
    assert lrc.metadata == (("ti", "示例"), ("作词", "作者"))
    assert line.text == "今日(きょう)、歌う"
    assert line.singer == "合"
    assert normalize_lyric_text(line.text) == "今日 歌う"
    assert translated.translation_for(line.id) == "Today, we sing"
    assert "[00:02.00]合: 今日(きょう)、歌う【Today, we sing】" in write_lrc(
        translated, include_translations=True
    )


def test_ttml_projects_fixture_lines_without_serializing_the_source(tmp_path: Path) -> None:
    fixture = tmp_path / "synthetic.ttml"
    fixture.write_text('<tt xmlns="http://www.w3.org/ns/ttml"><body><div>'
                       '<p key="L4" begin="1" end="2"><span begin="1" end="2">テスト</span></p>'
                       '<p key="L5" begin="3" end="4">二行目</p>'
                       '</div></body></tt>')
    document = read_ttml_for_alignment(fixture)
    assert len(document.lines) == 2
    assert document.lines[0].id == "L4"
    assert document.lines[0].text == "テスト"
    assert document.lines[0].source_start == 1.0


@pytest.fixture
def synthetic_audio(tmp_path: Path) -> Path:
    source = tmp_path / "generated.m4a"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:a", "aac", "-metadata", "title=夜奏花", "-metadata", "artist=YURiKA", str(source),
    ], check=True)
    return source


def test_ttml_rejects_unsupported_frame_time(tmp_path: Path) -> None:
    source = tmp_path / "unsupported.ttml"
    source.write_text('<tt><body><p key="L1" begin="12f" end="13f">歌</p></body></tt>')

    with pytest.raises(TtmlFormatError, match="不支持"):
        read_ttml_for_alignment(source)


def test_shared_similarity_and_search_score_handle_normalization_and_duration() -> None:
    fields = SearchScoreFields(180.0, ("鳥の詩 (TV Size)",), ("Lia",), ())

    assert text_similarity("今日(きょう)、歌う", "今日 歌う") == 1.0
    assert score_search_candidate(title="鳥の詩", duration=180.0, fields=fields, artist="Lia") > 75
    assert score_search_candidate(title="鳥の詩", duration=205.1, fields=fields, artist="Lia") == 0.0


def test_netease_parser_keeps_aliases_and_associates_translation() -> None:
    candidate = candidate_from_netease_result(
        {
            "id": 7,
            "name": "鳥の詩",
            "alia": ["鸟之诗"],
            "dt": 369_000,
            "ar": [{"name": "Lia", "alia": ["lia"]}],
            "al": {"name": "AIR", "tns": ["AIR 原声"]},
        },
        title="鳥の詩",
        duration=369.0,
        artist="Lia",
        album="AIR",
    )
    lyrics = lyrics_from_netease_payload(
        {"lrc": {"lyric": "[00:01.00]消える飛行機雲"}, "tlyric": {"lyric": "[00:01.00]消失的飞机云"}}
    )

    assert candidate.candidate_id == "7"
    assert candidate.match_score > 90
    assert lyrics is not None
    assert lyrics.translation_for("line-0") == "消失的飞机云"


def test_audio_metadata_reads_search_fields_without_returning_raw_tags(synthetic_audio: Path) -> None:
    metadata = read_audio_metadata(synthetic_audio)

    assert metadata.title == "夜奏花"
    assert metadata.artist == "YURiKA"
    assert metadata.duration == pytest.approx(2.0, abs=0.1)
    assert not hasattr(metadata, "raw_tags")


def test_ffmpeg_decode_uses_explicit_channel_sample_layout(synthetic_audio: Path) -> None:
    audio = decode_audio(synthetic_audio, sample_rate=8_000)

    assert audio.samples.dtype == np.float32
    assert audio.samples.shape[0] == 1
    assert audio.duration == pytest.approx(2.0, abs=0.1)


def test_activity_analysis_marks_core_and_padding_separately() -> None:
    samples = np.zeros((1, 10_000), dtype=np.float32); samples[:, 2_000:7_000] = 0.5
    analysis = analyze_vocals(AudioBuffer(samples, 1_000, "synthetic"), config=VocalConfig(min_segment_seconds=0.2))
    assert analysis.segments
    assert analysis.segments[0].start < analysis.segments[0].core_start


def test_resample_audio_preserves_duration_and_explicit_channel_count() -> None:
    audio = AudioBuffer(np.zeros((2, 4_410), dtype=np.float32), 44_100, "audio")
    converted = resample_audio(audio, sample_rate=16_000, channels=1)
    assert converted.samples.shape == (1, 1_600)
    assert converted.duration == pytest.approx(audio.duration)


def test_qq_parser_supports_base64_lyrics() -> None:
    candidate = candidate_from_qq_result({"mid": "abc", "name": "鳥の詩", "interval": 369, "singer": [{"name": "Lia"}], "album": {"name": "AIR"}}, title="鳥の詩", duration=369, artist="Lia", album=None)
    lyrics = lyrics_from_qq_payload({"code": 0, "lyric": "WzAwOjAxLjAwXea2iOWIpuOCi+mjnuihjOmdoumbsiJ9"})
    assert candidate.candidate_id == "abc"
    assert lyrics is not None


def test_lrclib_parser_falls_back_to_plain_text() -> None:
    lyrics = lyrics_from_lrclib_payload({"plainLyrics": "[ti:ignored]\n一行\n二行"})
    assert lyrics is not None
    assert [line.text for line in lyrics.document.lines] == ["一行", "二行"]


def test_public_lyric_search_fetch_and_rank_are_provider_independent(tmp_path: Path, synthetic_audio: Path) -> None:
    metadata = read_audio_metadata(synthetic_audio)
    provider = _FakeProvider()
    candidates = search_lyrics(metadata, providers=(provider,))
    fetched = fetch_lyrics(candidates, providers=(provider,))
    transcription = TranscriptionResult(
        BackendInfo("test", "test/model", "revision", "runtime", "float32"),
        (TranscriptionSegment("segment", "夜空を見上げて", "ja", None, None, None, None, (), {}),),
    )
    ranked = rank_lyrics(fetched, transcription, providers=(provider,))

    assert len(candidates) == 1
    assert candidates[0].match_score == 90.0
    assert fetched[0].lyrics.document.lines[0].text == "夜空を見上げて"
    assert ranked[0].content_score == 1.0


def test_ttml_safe_roles_ruby_untimed_whitespace_and_snapshot_ids(tmp_path):
    from lazulite.lyrics.ttml import read_ttml_for_japanese
    p=tmp_path/'ruby.ttml'
    p.write_text('<tt xmlns:tts="urn:style"><body><p begin="0ms"> <span tts:ruby="container"><span tts:ruby="base">海</span><span tts:ruby="text">うみ</span></span>  あ<span role="x-translation">sea</span><span role="x-bg">背景</span>!</p><p>かな</p></body></tt>')
    doc=read_ttml_for_alignment(p)
    ja=read_ttml_for_japanese(p)
    assert doc.lines[0].text==' 海  あ!'
    assert ja.lines[0].start_hint.time==0 and ja.lines[1].start_hint is None
    assert ja.lines[0].units[0].reading=='うみ'
    assert ja.lines[0].units[0].source_range==(1,2)
    assert read_ttml_for_japanese(p)==ja
    assert doc.lines[1].source_start is None


def test_ttml_time_units_and_reject_relative_containers(tmp_path):
    from lazulite.lyrics.ttml import _parse_time
    import pytest
    for value,expected in [('1.5',1.5),('1.5s',1.5),('1500ms',1.5),('1:02.5',62.5),('01:02:03.5',3723.5)]:
        assert _parse_time(value,field='test')==expected
    for value in ['NaN','2t','1:00:00:12','-1','inf']:
        with pytest.raises(TtmlFormatError):_parse_time(value,field='test')
    p=tmp_path/'relative.ttml';p.write_text('<tt><body begin="2s"><p>歌</p></body></tt>')
    with pytest.raises(TtmlFormatError):read_ttml_for_alignment(p)


def test_ttml_does_not_project_metadata_or_foreign_paragraphs(tmp_path):
    p=tmp_path/'namespaces.ttml'
    p.write_text('<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ext="urn:extension"><head><metadata><p>metadata</p></metadata></head><body><p key="good">あ<ext:span>annotation</ext:span>い</p><ext:p>foreign</ext:p><ext:wrapper><p>nested foreign</p></ext:wrapper></body></tt>')
    result=read_ttml_for_alignment(p)
    assert len(result.lines)==1 and result.lines[0].text=='あい'
