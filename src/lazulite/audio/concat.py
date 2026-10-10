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
class ConcatMap:
    sample_rate: int
    sample_count: int
    pieces: tuple[Piece, ...]

    @property
    def duration(self):
        return self.sample_count / self.sample_rate

    def __post_init__(self):
        if type(self.sample_rate) is not int or self.sample_rate <= 0 or self.sample_count < 0:
            raise ValueError('invalid concatenated sample layout')
        end = source_end = 0
        for p in self.pieces:
            if any(type(v) is not int for v in (p.source_start,p.source_end,p.concat_start,p.concat_end)):
                raise ValueError('piece coordinates must be integer samples')
            if not (source_end <= p.source_start < p.source_end and p.concat_start == end and
                    p.concat_end-p.concat_start == p.source_end-p.source_start):
                raise ValueError('invalid concat pieces')
            end, source_end = p.concat_end, p.source_end
        if end != self.sample_count:
            raise ValueError('pieces do not cover concatenated samples')

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
        if not 0 <= start < end <= self.duration + 1e-8:
            raise ValueError('interval outside concatenated waveform')
        rate = self.sample_rate
        result = []
        for i, p in enumerate(self.pieces):
            lo, hi = max(start * rate, p.concat_start), min(end * rate, p.concat_end)
            if hi > lo + 1e-6:
                result.append(Fragment((p.source_start + lo - p.concat_start)/rate,
                                       (p.source_start + hi - p.concat_start)/rate, i))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConcatAudio:
    audio: AudioBuffer
    pieces: tuple[Piece, ...]

    @property
    def mapping(self):
        return ConcatMap(self.audio.sample_rate, self.audio.samples.shape[1], self.pieces)

    def to_source(self, sample, *, endpoint):
        return self.mapping.to_source(sample, endpoint=endpoint)

    def to_concat(self, sample, *, endpoint='start'):
        return self.mapping.to_concat(sample, endpoint=endpoint)

    def fragments(self, start, end):
        return self.mapping.fragments(start, end)


def concatenate_cores(vocals: AudioBuffer, segments: tuple[VocalSegment, ...], *, sample_rate: int = 16000) -> ConcatAudio:
    mono = resample_audio(vocals, sample_rate=sample_rate, channels=1)
    intervals = []
    for s in segments:
        a, b = max(0, floor(s.core_start * sample_rate)), min(mono.samples.shape[1], ceil(s.core_end * sample_rate))
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
    return ConcatAudio(AudioBuffer(samples, sample_rate, vocals.source_id), tuple(pieces))
