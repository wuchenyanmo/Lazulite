"""无需下载模型的生成证据契约测试。"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from lazulite.alignment.forced.qwen import QwenForcedAligner
from lazulite.models.alignment import ForcedAlignmentRequest
from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.transcription.backends.qwen import QwenBackend
from lazulite.transcription.backends.whisper import WhisperBackend, _BYTE_DECODER
from lazulite.transcription.base import TranscriptionInput
from lazulite.transcription.token_scores import (
    export_full_logprobs,
    extract_token_scores,
)


def test_token_scores_use_full_distribution_without_renormalizing_top_k(tmp_path) -> None:
    summary = extract_token_scores(
        np.array([[0.0, 0.0, -1.0], [3.0, 0.0, -2.0]], dtype=np.float32),
        [1, 0],
        distribution_kind="raw_model",
        lexical_steps=[0, 1],
        top_k=2,
    )
    destination = export_full_logprobs(
        tmp_path / "distribution.npz",
        summary,
        [1, 0],
        lexical_steps=[0, 1],
        source_segment_id="segment-1",
        model_id="example",
        tokenizer_revision="revision",
    )

    assert summary.scores[0].logprob == pytest.approx(-0.8619948)
    assert np.exp(summary.scores[0].top_tokens[0].logprob) < 1.0
    assert summary.confidence is not None
    with np.load(destination) as archive:
        assert archive["logprobs"].shape == (2, 3)
        assert archive["generated_ids"].tolist() == [1, 0]


def test_empty_lexical_range_has_no_fake_confidence() -> None:
    summary = extract_token_scores(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [0],
        distribution_kind="generation_processed",
        lexical_steps=[],
    )

    assert summary.scores == ()
    assert summary.confidence is None


def test_forced_output_maps_repeated_text_sequentially_and_keeps_zero_times() -> None:
    request = ForcedAlignmentRequest("L1", "ああい", "ja", 10.0, (0, 3), "word")
    units, diagnostics = QwenForcedAligner._map_units(
        (
            SimpleNamespace(text="あ", start_time=0.0, end_time=0.0),
            SimpleNamespace(text="あ", start_time=0.1, end_time=0.2),
            SimpleNamespace(text="い", start_time=0.2, end_time=0.3),
            SimpleNamespace(text="う", start_time=0.3, end_time=0.4),
        ),
        request,
    )

    assert [unit.input_range for unit in units] == [(0, 1), (1, 2), (2, 3)]
    assert units[0].start == 10.0
    assert units[0].end == 10.0
    assert all(unit.granularity == "character" for unit in units)
    assert diagnostics == ("unmapped:う",)


class _Tensor:
    def __init__(self, values): self._values = values
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self._values


class _WhisperTokenizer:
    all_special_ids: set[int] = set()

    def __init__(self, tokens): self._tokens = tokens
    def convert_ids_to_tokens(self, token_id): return self._tokens[token_id]


def test_whisper_unicode_group_uses_its_first_token_timestamp() -> None:
    reverse = {value: key for key, value in _BYTE_DECODER.items()}
    first, second = "漢".encode("utf-8")[:2], "漢".encode("utf-8")[2:]
    tokens = {1: "".join(reverse[value] for value in first), 2: "".join(reverse[value] for value in second), 3: "A"}
    backend = WhisperBackend.__new__(WhisperBackend)
    backend._processor = SimpleNamespace(tokenizer=_WhisperTokenizer(tokens))
    item = TranscriptionInput(AudioBuffer(np.zeros((1, 176_000), dtype=np.float32), 16_000, "audio"), VocalSegment("s", None, 10, 11, 10, 11, None, None))

    timed = backend._timed_text([1, 2, 3], 1, [_Tensor([0.0, 1.0, 2.0, 3.0, 4.0])], item)

    assert [(unit.text, unit.start, unit.end) for unit in timed] == [("漢", 11.0, 13.0), ("A", 13.0, None)]


class _QwenTokenizer:
    all_special_ids = {99}
    _tokens = {1: "language Japanese", 2: "<asr_text>", 3: "歌", 99: ""}

    def batch_decode(self, values, **kwargs):
        return ["".join(self._tokens[item] for item in values[0] if item not in self.all_special_ids)]


def test_qwen_automatic_language_header_is_not_counted_as_lyric_tokens() -> None:
    language, text, steps = QwenBackend._body_output(
        _QwenTokenizer(), [1, 2, 3, 99], forced_language=None,
        parse_asr_output=lambda raw, user_language: ("Japanese", "歌"),
    )

    assert language == "Japanese"
    assert text == "歌"
    assert steps == [2]
