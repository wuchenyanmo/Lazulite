"""Sample-exact core union and sided mapping, independent of lyrics/models."""
from dataclasses import dataclass
from math import ceil, floor, isfinite
import numpy as np
from lazulite.models.audio import AudioBuffer, VocalSegment
from lazulite.models.japanese import Fragment
from .io import resample_audio


@dataclass(frozen=True, slots=True)
class Piece:
    source_start: int
    source_end: int
    concat_start: int
    concat_end: int


@dataclass(frozen=True, slots=True)
class ConcatAudio:
    audio: AudioBuffer
    pieces: tuple[Piece, ...]

    def to_source(self, sample: float, *, endpoint: str) -> float | None:
        if endpoint not in {'start', 'end'} or not isfinite(sample):
            raise ValueError('invalid endpoint/sample')
        for p in self.pieces:
            inside = p.concat_start <= sample < p.concat_end if endpoint == 'start' else p.concat_start < sample <= p.concat_end
            if inside:
                return p.source_start + sample - p.concat_start
        return None

    def to_concat(self, sample: float, *, endpoint: str = 'start') -> float | None:
        if endpoint not in {'start', 'end'} or not isfinite(sample):
            raise ValueError('invalid endpoint/sample')
        for p in self.pieces:
            inside = p.source_start <= sample < p.source_end if endpoint == 'start' else p.source_start < sample <= p.source_end
            if inside:
                return p.concat_start + sample - p.source_start
        return None

    def fragments(self, start: float, end: float) -> tuple[Fragment, ...]:
        if not 0 <= start < end <= self.audio.duration + 1e-8:
            raise ValueError('interval outside concatenated waveform')
        rate = self.audio.sample_rate
        result = []
        for i, p in enumerate(self.pieces):
            lo, hi = max(start * rate, p.concat_start), min(end * rate, p.concat_end)
            if hi > lo + 1e-6:
                result.append(Fragment((p.source_start + lo - p.concat_start)/rate,
                                       (p.source_start + hi - p.concat_start)/rate, i))
        return tuple(result)


def concatenate_cores(vocals: AudioBuffer, segments: tuple[VocalSegment, ...]) -> ConcatAudio:
    mono = resample_audio(vocals, sample_rate=16000, channels=1)
    intervals = []
    for s in segments:
        a, b = max(0, floor(s.core_start * 16000)), min(mono.samples.shape[1], ceil(s.core_end * 16000))
        if b > a:
            intervals.append((a,b))
    merged = []
    for a,b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a,b])
    offset, pieces = 0, []
    for a,b in merged:
        pieces.append(Piece(a,b,offset,offset+b-a))
        offset += b-a
    samples = np.concatenate([mono.samples[:,a:b] for a,b in merged], axis=1) if merged else np.empty((1,0),np.float32)
    return ConcatAudio(AudioBuffer(samples, 16000, vocals.source_id), tuple(pieces))
