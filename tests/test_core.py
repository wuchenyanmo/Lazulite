"""不依赖模型栈的核心类型回归测试。"""

from __future__ import annotations

import importlib
import sys
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
    assert "[00:02.00]今日(きょう)、歌う【Today, we sing】" in write_lrc(
        translated, include_translations=True
    )


@pytest.mark.parametrize(
    ("filename", "line_count", "first_id", "first_text"),
    [
        ("POWDER SNOW (Live Ver.).ttml", 35, "L4", "粉雪が空から"),
        ("夜奏花.ttml", 33, "L4", "夜空を見上げて"),
        ("鳥の詩.ttml", 42, "L4", "消える飛行機雲 僕たちは見送った"),
    ],
)
def test_ttml_projects_fixture_lines_without_serializing_the_source(
    filename: str, line_count: int, first_id: str, first_text: str
) -> None:
    fixture = Path(__file__).parent / "fixtures" / filename
    document = read_ttml_for_alignment(fixture)

    assert len(document.lines) == line_count
    assert document.lines[0].id == first_id
    assert document.lines[0].text == first_text
    assert document.lines[0].source_start is not None


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


def test_audio_metadata_reads_search_fields_without_returning_raw_tags() -> None:
    metadata = read_audio_metadata(Path(__file__).parent / "fixtures" / "夜奏花.m4a")

    assert metadata.title == "夜奏花"
    assert metadata.artist == "YURiKA"
    assert metadata.duration == pytest.approx(353.003, abs=0.1)
    assert not hasattr(metadata, "raw_tags")


def test_ffmpeg_decode_uses_explicit_channel_sample_layout() -> None:
    audio = decode_audio(Path(__file__).parent / "fixtures" / "夜奏花.m4a", sample_rate=8_000)

    assert audio.samples.dtype == np.float32
    assert audio.samples.shape[0] == 1
    assert audio.duration == pytest.approx(353.003, abs=0.1)


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


def test_public_lyric_search_fetch_and_rank_are_provider_independent(tmp_path: Path) -> None:
    metadata = read_audio_metadata(Path(__file__).parent / "fixtures" / "夜奏花.m4a")
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
