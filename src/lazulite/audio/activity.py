"""Two-stage vocal activity, valley segmentation and track-level evidence."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment


@dataclass(frozen=True, slots=True)
class VocalConfig:
    frame_length: int = 2048
    hop_length: int = 256
    smoothing_seconds: float = 0.10
    enter_threshold: float = 0.52
    keep_threshold: float = 0.30
    padding_seconds: float = 0.18
    min_segment_seconds: float = 0.45
    max_gap_seconds: float = 0.22
    max_segment_seconds: float = 8.0
    long_valley_threshold: float = 0.30
    vocal_db_threshold: float = -30.0
    min_vocal_seconds: float = 5.0
    min_vocal_ratio: float = 0.20

    def __post_init__(self):
        if self.frame_length < 2 or self.hop_length < 1:
            raise ValueError("frame_length/hop_length must be positive")
        if not 0 <= self.keep_threshold <= self.enter_threshold <= 1:
            raise ValueError("invalid hysteresis thresholds")
        if not 0 < self.min_segment_seconds < self.max_segment_seconds:
            raise ValueError("invalid segment durations")
        if (
            min(
                self.smoothing_seconds,
                self.padding_seconds,
                self.max_gap_seconds,
                self.min_vocal_seconds,
            )
            < 0
            or not 0 <= self.min_vocal_ratio <= 1
        ):
            raise ValueError("invalid vocal configuration")


def _intervals(mask):
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _smooth(values, size):
    size = max(1, int(size))
    if not len(values):
        return values
    # Preserve frame count even when the smoothing kernel exceeds a short clip.
    result = np.convolve(values, np.ones(size, dtype=np.float32) / size, mode="full")
    return result[(size - 1) // 2 : (size - 1) // 2 + len(values)]


def _rms(values, frame_length=2048, hop_length=256):
    padded = np.pad(values, (frame_length // 2, frame_length // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[
        ::hop_length
    ]
    return np.sqrt(np.mean(frames * frames, axis=1))


def _db(values, reference):
    db = 20 * np.log10(np.maximum(values, 1e-5)) - 20 * np.log10(max(reference, 1e-5))
    return np.maximum(db, db.max() - 80)


def main_vocal_audio(audio: AudioBuffer) -> AudioBuffer:
    if audio.samples.shape[0] == 1:
        return audio
    if audio.samples.shape[0] != 2:
        raise ValueError("vocal analysis requires mono or stereo")
    left, right = audio.samples
    main = (left + right) / 2 - np.abs((left - right) / 2)
    return AudioBuffer(
        main[None, :].astype(np.float32), audio.sample_rate, audio.source_id
    )


def build_activity_profile(audio: AudioBuffer, *, config: VocalConfig = VocalConfig()):
    main = main_vocal_audio(audio).samples[0]
    mid = audio.samples.mean(axis=0)
    stereo = audio.samples.shape[0] == 2
    side = (audio.samples[0] - audio.samples[1]) / 2 if stereo else np.zeros_like(mid)
    rms = _rms(main, config.frame_length, config.hop_length)
    db = _db(rms, max(float(rms.max()), 1e-6))
    floor, ceiling = np.percentile(db, (5, 95))
    energy = np.clip((db - floor) / max(ceiling - floor, 1), 0, 1)
    mid_rms = _rms(mid, config.frame_length, config.hop_length)
    side_rms = _rms(np.abs(side), config.frame_length, config.hop_length)
    ratio = mid_rms / (mid_rms + side_rms + 1e-8)
    low, high = np.percentile(ratio, (10, 90))
    center = (
        np.clip((ratio - low) / max(high - low, 1e-3), 0, 1)
        if stereo
        else np.ones_like(ratio)
    )
    frames = lambda seconds: max(
        round(seconds * audio.sample_rate / config.hop_length), 1
    )
    center_smooth = np.clip(
        _smooth(center, frames(max(config.smoothing_seconds * 2.5, 0.18))), 0, 1
    )
    center_long = np.clip(
        _smooth(center, frames(max(0.85, config.smoothing_seconds * 10))), 0, 1
    )
    activity = _smooth(
        energy * (0.35 + 0.65 * center_smooth), frames(config.smoothing_seconds)
    )
    return {
        "main_rms": rms,
        "main_db": db,
        "energy_score": energy,
        "center_score_long": center_long,
        "activity": activity,
        "split_activity": _smooth(
            activity, frames(max(0.30, config.smoothing_seconds * 4))
        ),
        "frame_times": np.arange(len(activity)) * config.hop_length / audio.sample_rate,
    }


def vocal_evidence(
    audio: AudioBuffer,
    *,
    reference: AudioBuffer | None = None,
    config: VocalConfig = VocalConfig(),
):
    main = main_vocal_audio(audio).samples[0]
    if not main.size:
        return False, 0.0
    ref = (
        1.0
        if reference is None
        else max(float(np.max(np.abs(reference.samples.mean(axis=0)))), 1e-6)
    )
    rms = _rms(main, 2048, 512)
    voiced_seconds = float(
        np.count_nonzero(_db(rms, ref) > config.vocal_db_threshold)
        * 512
        / audio.sample_rate
    )
    duration = len(rms) * 512 / audio.sample_rate
    return (
        voiced_seconds >= config.min_vocal_seconds
        and voiced_seconds >= config.min_vocal_ratio * duration,
        voiced_seconds,
    )


def split_long_interval(
    config,
    start_frame: int,
    end_frame: int,
    activity: np.ndarray,
    split_activity: np.ndarray,
    sr: int,
) -> list[tuple[int, int]]:
    from scipy.signal import find_peaks

    intervals = [(start_frame, end_frame)]
    min_frames = max(int(round(config.min_segment_seconds * sr / config.hop_length)), 1)
    max_frames = max(
        int(round(config.max_segment_seconds * sr / config.hop_length)), min_frames + 1
    )
    valley_window = max(int(round(0.18 * sr / config.hop_length)), 1)
    min_peak_distance = max(min_frames // 2, 1)
    while True:
        next_intervals = []
        changed = False
        for cur_start, cur_end in intervals:
            if cur_end - cur_start <= max_frames:
                next_intervals.append((cur_start, cur_end))
                continue
            local_split = split_activity[cur_start:cur_end]
            if local_split.size <= 2 * min_frames:
                next_intervals.append((cur_start, cur_end))
                continue
            search_local = local_split[min_frames : len(local_split) - min_frames]
            best_split = None
            best_score = float("-inf")
            if search_local.size > 0:
                dynamic_range = float(
                    np.percentile(local_split, 85) - np.percentile(local_split, 15)
                )
                min_prominence = max(0.035, 0.22 * dynamic_range)
                valley_indices, valley_props = find_peaks(
                    -search_local, prominence=min_prominence, distance=min_peak_distance
                )
                if valley_indices.size > 0:
                    prominences = valley_props.get(
                        "prominences", np.zeros_like(valley_indices, dtype=np.float32)
                    )
                    interval_median = float(np.median(local_split))
                    interval_low = float(np.percentile(local_split, 20))
                    for local_idx, prominence in zip(valley_indices, prominences):
                        frame_idx = cur_start + min_frames + int(local_idx)
                        local_start = max(cur_start, frame_idx - valley_window)
                        local_end = min(cur_end, frame_idx + valley_window)
                        valley_value = float(
                            np.mean(split_activity[local_start:local_end])
                        )
                        raw_value = float(np.mean(activity[local_start:local_end]))
                        valley_depth = interval_median - valley_value
                        low_bonus = max(0.0, interval_low - valley_value)
                        distance_penalty = abs(
                            frame_idx - (cur_start + cur_end) / 2
                        ) / max(cur_end - cur_start, 1)
                        raw_bonus = max(0.0, interval_median - raw_value)
                        candidate_score = (
                            1.35 * float(prominence)
                            + 0.55 * valley_depth
                            + 0.35 * low_bonus
                            + 0.2 * raw_bonus
                            - 0.18 * distance_penalty
                        )
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_split = frame_idx
            if best_split is None or best_score < config.long_valley_threshold:
                hard_split = cur_start + max_frames
                if cur_end - hard_split < min_frames:
                    hard_split = (cur_start + cur_end) // 2
                best_split = hard_split
            next_intervals.append((cur_start, best_split))
            next_intervals.append((best_split, cur_end))
            changed = True
        intervals = next_intervals
        if not changed:
            break
    return intervals


def _initial_segment_intervals(
    config, activity: np.ndarray, sr: int
) -> list[tuple[int, int]]:
    high_mask = activity >= config.enter_threshold
    low_mask = activity >= config.keep_threshold
    candidate_intervals = []
    for high_start, high_end in _intervals(high_mask):
        left = high_start
        right = high_end
        while left > 0 and low_mask[left - 1]:
            left -= 1
        while right < len(low_mask) and low_mask[right]:
            right += 1
        candidate_intervals.append((left, right))
    if not candidate_intervals:
        return []
    merged = [candidate_intervals[0]]
    max_gap_frames = int(round(config.max_gap_seconds * sr / config.hop_length))
    for start, end in candidate_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_frames:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    min_frames = int(round(config.min_segment_seconds * sr / config.hop_length))
    return [(start, end) for start, end in merged if end - start >= max(min_frames, 1)]


def analyze_vocals(
    audio: AudioBuffer,
    *,
    config: VocalConfig = VocalConfig(),
    reference: AudioBuffer | None = None,
) -> VocalAnalysis:
    if audio.samples.shape[1] == 0:
        return VocalAnalysis(audio.source_id, (), ("no_vocal_evidence",), False, 0.0)
    profile = build_activity_profile(audio, config=config)
    active, voiced_seconds = vocal_evidence(audio, reference=reference, config=config)
    activity, split = profile["activity"], profile["split_activity"]
    segments = []
    for section, (start, end) in enumerate(
        _initial_segment_intervals(config, activity, audio.sample_rate), 1
    ):
        pieces = split_long_interval(
            config, start, end, activity, split, audio.sample_rate
        )
        for first, last in pieces:
            core_start = min(first * config.hop_length, audio.samples.shape[1])
            core_end = min(last * config.hop_length, audio.samples.shape[1])
            if core_end <= core_start:
                continue
            selected = slice(first, last)
            presence = float(
                np.clip(
                    0.62 * np.mean(activity[selected])
                    + 0.38 * np.mean(profile["energy_score"][selected]),
                    0,
                    1,
                )
            )
            center = profile["center_score_long"][selected]
            risk = (
                float(
                    presence
                    * np.clip(
                        0.62 * (1 - np.percentile(center, 80))
                        + 0.38 * np.mean(center < 0.35),
                        0,
                        1,
                    )
                )
                if audio.samples.shape[0] == 2
                else None
            )
            a, b = core_start / audio.sample_rate, core_end / audio.sample_rate
            segments.append(
                VocalSegment(
                    f"vocal-{len(segments)}",
                    f"section-{section}",
                    max(0, a - config.padding_seconds),
                    min(audio.duration, b + config.padding_seconds),
                    a,
                    b,
                    presence,
                    risk,
                )
            )
    diagnostics = []
    if not active or not segments:
        diagnostics.append("no_vocal_evidence")
    if reference is None:
        diagnostics.append("vocal_reference_full_scale")
    if audio.samples.shape[0] == 1:
        diagnostics.append("missing_stereo_evidence")
    return VocalAnalysis(
        audio.source_id, tuple(segments), tuple(diagnostics), active, voiced_seconds
    )
