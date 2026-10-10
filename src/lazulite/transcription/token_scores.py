"""生成分布的归一化、gather、top-k 与显式 NPZ 导出。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

DistributionKind = Literal["generation_processed", "raw_model"]


@dataclass(frozen=True, slots=True)
class TopToken:
    token_id: int
    logprob: float


@dataclass(frozen=True, slots=True)
class TokenScore:
    generation_step: int
    token_id: int
    logprob: float
    top_tokens: tuple[TopToken, ...]


@dataclass(frozen=True, slots=True)
class TokenScoreSummary:
    distribution_kind: DistributionKind
    scores: tuple[TokenScore, ...]
    confidence: float | None
    logprobs: NDArray[np.float32]

    def metadata(self) -> dict[str, object]:
        """生成可写入常规 JSON 的小型证据，不包含完整词表矩阵。"""
        return {
            "distribution_kind": self.distribution_kind,
            "token_scores": [
                {
                    "generation_step": score.generation_step,
                    "token_id": score.token_id,
                    "logprob": score.logprob,
                    "top_tokens": [
                        {"token_id": top.token_id, "logprob": top.logprob}
                        for top in score.top_tokens
                    ],
                }
                for score in self.scores
            ],
        }


def normalized_logprobs(distributions: NDArray[np.floating]) -> NDArray[np.float32]:
    """在完整词表上计算 float32 log-softmax，绝不对 top-k 再归一化。"""
    values = np.asarray(distributions, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("生成分布必须为非空 [steps, vocab] 矩阵")
    row_max = np.max(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(row_max)):
        raise ValueError("生成分布至少应在每一步包含一个有限值")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = values - row_max
        log_normalizer = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return (shifted - log_normalizer).astype(np.float32, copy=False)


def extract_token_scores(
    distributions: NDArray[np.floating],
    generated_ids: Sequence[int],
    *,
    distribution_kind: DistributionKind,
    lexical_steps: Sequence[int] | None = None,
    top_k: int = 5,
) -> TokenScoreSummary:
    """从同一次生成的每步分布提取真实 token 分数和可选 top-k。"""
    if top_k < 0:
        raise ValueError("top_k 不能为负数")
    logprobs = normalized_logprobs(distributions)
    generated = np.asarray(generated_ids, dtype=np.int64)
    if generated.ndim != 1 or len(generated) != logprobs.shape[0]:
        raise ValueError("generated_ids 必须与生成步数一一对应")
    if np.any(generated < 0) or np.any(generated >= logprobs.shape[1]):
        raise ValueError("generated_ids 包含超出词表范围的 token")
    steps = tuple(lexical_steps) if lexical_steps is not None else tuple(range(len(generated)))
    if len(set(steps)) != len(steps) or any(step < 0 or step >= len(generated) for step in steps):
        raise ValueError("lexical_steps 必须是唯一且有效的生成步下标")

    scores: list[TokenScore] = []
    for step in steps:
        token_id = int(generated[step])
        selected = float(logprobs[step, token_id])
        if not np.isfinite(selected):
            raise ValueError("正文 token 的选中 logprob 必须为有限值")
        top_tokens: tuple[TopToken, ...] = ()
        if top_k:
            finite_indices = np.flatnonzero(np.isfinite(logprobs[step]))
            selected_indices = finite_indices[
                np.argsort(logprobs[step, finite_indices])[::-1][:top_k]
            ]
            top_tokens = tuple(
                TopToken(token_id=int(index), logprob=float(logprobs[step, index]))
                for index in selected_indices
            )
        scores.append(TokenScore(step, token_id, selected, top_tokens))
    confidence = exp(float(np.mean([score.logprob for score in scores]))) if scores else None
    return TokenScoreSummary(distribution_kind, tuple(scores), confidence, logprobs)


def export_full_logprobs(
    path: str | Path,
    summary: TokenScoreSummary,
    generated_ids: Sequence[int],
    *,
    lexical_steps: Sequence[int],
    source_segment_id: str,
    model_id: str,
) -> Path:
    """显式保存可恢复的完整分布；常规 JSON 只保存该文件引用和摘要。"""
    if not source_segment_id or not model_id:
        raise ValueError("全分布导出必须记录分片、模型和 tokenizer 身份")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        logprobs=summary.logprobs,
        generated_ids=np.asarray(generated_ids, dtype=np.int64),
        lexical_steps=np.asarray(lexical_steps, dtype=np.int64),
        distribution_kind=np.asarray(summary.distribution_kind),
        source_segment_id=np.asarray(source_segment_id),
        model_id=np.asarray(model_id),
    )
    return target
