"""无需下载模型的生成证据契约测试。"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from lazulite.alignment.forced.qwen import QwenForcedAligner
from lazulite.models.alignment import ForcedAlignmentRequest
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
