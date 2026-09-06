"""不依赖模型栈的核心类型回归测试。"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.lyrics import LyricDocument, LyricLine, ReadingSpan
from lazulite.models.transcription import TranscriptionSegment
from lazulite.lyrics.lrc import read_lrc, with_translation, write_lrc
from lazulite.lyrics.normalize import normalize_lyric_text


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
