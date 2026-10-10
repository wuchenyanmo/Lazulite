"""MiliKara fixed activity algorithm, adapted to original-axis core concatenation.

MIT ff2d2f54706ecb140bc5b275037eff64187ef6ef; see THIRD_PARTY_LICENSES.
RMS and smoothing run BEFORE cutting cores. Frame windows use the decoder's
20 ms stride, with sample-weighted original envelope integration across seams.
"""
import numpy as np
from lazulite.audio.io import resample_audio

VERSION = 'milikara-rms30-hop10-smooth300-core-integral-v1'


def original_rest(audio):
    mono = resample_audio(audio, sample_rate=16000, channels=1).samples[0].astype(np.float64)
    hop, win = 160, 480
    count = (len(mono) + hop - 1) // hop
    if not count:
        return np.empty(0, np.float32), None
    squared = np.pad(mono * mono, (win // 2, win))
    cs = np.r_[0., np.cumsum(squared)]
    starts = np.arange(count) * hop
    rms = np.sqrt(np.maximum((cs[starts + win] - cs[starts]) / win, 0))
    env = np.maximum(20 * np.log10(np.maximum(rms, 1e-12)), -120).astype(np.float32)
    power = np.power(10., env.astype(np.float64) / 10.)
    # 'same' on normal songs; keep input length even below the 300 ms window.
    smooth = np.convolve(power, np.ones(30) / 30, mode='full')[14:14 + count]
    level = 10 * np.log10(np.maximum(smooth, 1e-12))
    audible = level[level > -80]
    ref = float(np.percentile(audible, 90)) if len(audible) else None
    if ref is None or ref < -60:
        return np.zeros(count, np.float32), ref
    return np.clip((ref - 8 - level) / 12, 0, 1).astype(np.float32), ref


def map_rest(rest, concat, frames, frame_map):
    # Envelope bin i owns [i*160, (i+1)*160). Integrate fractional bins,
    # including a frame crossing a cut, without smoothing across that cut.
    cumulative = np.r_[0., np.cumsum(rest, dtype=np.float64)] * 160
    def integral(samples):
        indices = np.floor(samples / 160).astype(np.int64)
        return cumulative[indices] + (samples - indices * 160) * rest[np.minimum(indices, len(rest)-1)]
    scale = 16000 / concat.sample_rate
    stride = frame_map.seconds * 16000
    lo = (frame_map.offset_samples + np.arange(frames) * frame_map.stride_samples) * scale
    hi = lo + stride
    values = np.zeros(frames, np.float64)
    for p in concat.pieces:
        a, b = np.maximum(lo, p.concat_start * scale), np.minimum(hi, p.concat_end * scale)
        inside = b > a
        offset = (p.source_start - p.concat_start) * scale
        values[inside] += integral(b[inside] + offset) - integral(a[inside] + offset)
    return np.clip(values / stride, 0, 1)


def frame_activity(prepared, concat, frames, frame_map):
    rest, reference = original_rest(prepared.vocals)
    return map_rest(rest, concat, frames, frame_map), {
        'algorithm': VERSION, 'sample_rate': 16000,
        'rms_window_samples': 480, 'envelope_hop_samples': 160,
        'smoothing_ms': 300, 'reference_db': reference,
        'frame_stride_seconds': frame_map.seconds, 'frame_offset_seconds': frame_map.time(0),
        'mapping': 'original envelope integral over concat pieces; no seam smoothing',
        'silent_policy': 'unavailable reference; no rest penalties',
        'available': reference is not None and reference >= -60,
    }
