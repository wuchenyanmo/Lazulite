"""从已分离人声中生成可追溯声学窗口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment


@dataclass(frozen=True, slots=True)
class VocalConfig:
    frame_seconds: float = 0.032
    hop_seconds: float = 0.016
    enter_threshold: float = 0.52
    keep_threshold: float = 0.30
    padding_seconds: float = 0.18
    min_segment_seconds: float = 0.45
    max_gap_seconds: float = 0.22
    max_segment_seconds: float = 8.0


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def analyze_vocals(audio: AudioBuffer, *, config: VocalConfig = VocalConfig()) -> VocalAnalysis:
    """迁移旧声学规则的核心：能量门控、滞回、短停顿合并和最长窗口拆分。"""
    mono = audio.samples.mean(axis=0)
    frame = max(1, round(config.frame_seconds * audio.sample_rate)); hop = max(1, round(config.hop_seconds * audio.sample_rate))
    if len(mono) < frame:
        return VocalAnalysis(audio.source_id, (), ("no_vocal_evidence",))
    windows = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    rms = np.sqrt(np.mean(windows * windows, axis=1))
    low, high = np.percentile(rms, (5, 95)); activity = np.clip((rms - low) / max(high - low, 1e-8), 0, 1)
    candidates: list[tuple[int, int]] = []
    for start, end in _intervals(activity >= config.enter_threshold):
        while start > 0 and activity[start - 1] >= config.keep_threshold: start -= 1
        while end < len(activity) and activity[end] >= config.keep_threshold: end += 1
        candidates.append((start, end))
    merged: list[tuple[int, int]] = []
    gap = round(config.max_gap_seconds * audio.sample_rate / hop)
    for start, end in candidates:
        if merged and start - merged[-1][1] <= gap: merged[-1] = (merged[-1][0], end)
        else: merged.append((start, end))
    segments: list[VocalSegment] = []; minimum = config.min_segment_seconds * audio.sample_rate
    maximum = config.max_segment_seconds * audio.sample_rate; padding = config.padding_seconds * audio.sample_rate
    for section, (start_frame, end_frame) in enumerate(merged, 1):
        core_start, core_end = start_frame * hop, min(len(mono), end_frame * hop + frame)
        for piece_start in np.arange(core_start, core_end, maximum, dtype=int):
            piece_end = min(core_end, int(piece_start + maximum))
            if piece_end - piece_start < minimum: continue
            frame_start, frame_end = max(0, piece_start // hop), min(len(activity), int(np.ceil(piece_end / hop)))
            presence = float(np.mean(activity[frame_start:frame_end]))
            segments.append(VocalSegment(f"vocal-{len(segments)}", f"section-{section}", max(0.0, (piece_start-padding)/audio.sample_rate), min(audio.duration, (piece_end+padding)/audio.sample_rate), piece_start/audio.sample_rate, piece_end/audio.sample_rate, presence, None))
    diagnostics = () if segments else ("no_vocal_evidence",)
    return VocalAnalysis(audio.source_id, tuple(segments), diagnostics)
