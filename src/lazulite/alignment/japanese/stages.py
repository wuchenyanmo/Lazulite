"""Public numerical and coordinate contracts for reusable Japanese stages."""
from dataclasses import dataclass, field, asdict
from math import isfinite
import numpy as np
from lazulite.audio.concat import ConcatMap
from lazulite.models.japanese import JapaneseAlignmentRequest


@dataclass(frozen=True, slots=True)
class FrameMap:
    sample_rate: int
    stride_samples: int
    receptive_samples: int
    offset_samples: float = 0

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.sample_rate,self.stride_samples,self.receptive_samples)) or not isfinite(self.offset_samples) or self.offset_samples < 0:
            raise ValueError('invalid frame mapping')

    @property
    def seconds(self):
        return self.stride_samples / self.sample_rate

    def time(self, frame):
        return self.offset_samples/self.sample_rate + frame*self.seconds

    def index(self, seconds):
        return (seconds*self.sample_rate-self.offset_samples)/self.stride_samples

    def count(self, samples):
        return max(0, (samples-self.receptive_samples)//self.stride_samples+1)


@dataclass(frozen=True, slots=True)
class AcousticResult:
    log_probabilities: np.ndarray
    vocabulary: dict[str, int]
    blank_id: int
    frames: FrameMap
    mapping: ConcatMap
    original_duration: float
    rest: np.ndarray | None = None
    description: dict = field(default_factory=dict)
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self):
        self.frames.__post_init__()
        self.mapping.__post_init__()
        lp = self.log_probabilities
        if lp.ndim != 2 or not np.issubdtype(lp.dtype,np.floating) or np.isnan(lp).any() or np.isposinf(lp).any() or np.any(lp > 1e-5):
            raise ValueError('expected frame log probabilities')
        if lp.shape[1] == 0 or type(self.blank_id) is not int or not 0 <= self.blank_id < lp.shape[1]:
            raise ValueError('invalid vocabulary/blank')
        ids = list(self.vocabulary.values())
        if not ids or any(type(v) is not int or not 0 <= v < lp.shape[1] for v in ids) or len(set(ids)) != len(ids) or any(not isinstance(k,str) or not k for k in self.vocabulary):
            raise ValueError('invalid token columns')
        if len(lp) and not np.allclose(np.logaddexp.reduce(lp,axis=1,dtype=np.float64),0,atol=2e-4):
            raise ValueError('rows must be normalized log probabilities')
        if self.frames.sample_rate != self.mapping.sample_rate or (len(lp) and self.frames.time(len(lp)) > self.mapping.duration+1e-8):
            raise ValueError('frame mapping outside concatenated audio')
        if not isfinite(self.original_duration) or self.original_duration < 0 or any(p.source_end/self.mapping.sample_rate > self.original_duration+1e-8 for p in self.mapping.pieces):
            raise ValueError('pieces outside original duration')
        if self.rest is not None and (self.rest.shape != (len(lp),) or not np.isfinite(self.rest).all() or np.any((self.rest<0)|(self.rest>1))):
            raise ValueError('rest must contain one finite 0–1 value per frame')


@dataclass(frozen=True, slots=True)
class DecodeResult:
    request: JapaneseAlignmentRequest
    tokens: tuple[int, ...]
    vocabulary: dict[str, int]
    blank_id: int
    special_tokens: tuple[str, ...]
    vocabulary_size: int
    ranges: dict[str, tuple[int, int]]
    texts: dict[str, str]
    problems: dict[str, str]
    first_pass: tuple[tuple[int,int], ...] | None
    spans: tuple[tuple[int,int], ...] | None
    frame_count: int
    frames: FrameMap
    mapping: ConcatMap
    original_duration: float
    execution: dict
    calibration: dict
    reason: str | None = None

    def __post_init__(self):
        self.request.__post_init__()
        self.frames.__post_init__()
        self.mapping.__post_init__()
        if (type(self.frame_count) is not int or self.frame_count < 0 or
            self.frames.sample_rate != self.mapping.sample_rate or
            self.frame_count and self.frames.time(self.frame_count)>self.mapping.duration+1e-8 or
            not isfinite(self.original_duration) or self.original_duration<0 or
            any(p.source_end/self.mapping.sample_rate>self.original_duration+1e-8 for p in self.mapping.pieces)):
            raise ValueError('invalid decoded coordinates')
        if (type(self.blank_id) is not int or not 0<=self.blank_id<self.vocabulary_size or
            any(type(v) is not int or not 0<=v<self.vocabulary_size for v in self.vocabulary.values()) or
            any(t not in self.vocabulary.values() or t==self.blank_id for t in self.tokens)):
            raise ValueError('invalid decoded target IDs')
        for key in ('first_pass_score','second_pass_score'):
            if key in self.execution and not isfinite(self.execution[key]): raise ValueError('invalid decode score')
        ids = [u.unit_id for l in self.request.lines for u in l.units]
        if set(ids) != set(self.ranges) or set(ids) != set(self.texts) or set(self.problems)-set(ids):
            raise ValueError('decoded target association mismatch')
        end=0
        for uid in ids:
            a,b=self.ranges[uid]
            if a!=end or not a<=b<=len(self.tokens): raise ValueError('invalid target ranges')
            end=b
        if end!=len(self.tokens) or any(type(t) is not int or t<0 for t in self.tokens):
            raise ValueError('invalid decoded targets')
        if self.execution.get('input') != asdict(self.request):
            raise ValueError('decoded execution and target request differ')
        if self.execution.get('profile') == 'ja-hepburn/2':
            from .run import _profile
            texts,tokens,ranges,problems = _profile(self.request,self.vocabulary,self.blank_id,self.special_tokens)
            if (texts,tuple(tokens),ranges,problems)!=(self.texts,self.tokens,self.ranges,self.problems):
                raise ValueError('decoded targets do not match readings')
        if (self.spans is None)!=(self.first_pass is None) or (self.spans is None and not self.reason):
            raise ValueError('decoded path status mismatch')
        for spans in (self.first_pass,self.spans):
            if spans is not None:
                if len(spans)!=len(self.tokens) or any(type(a) is not int or type(b) is not int or not 0<=a<b<=self.frame_count for a,b in spans) or any(x[1]>y[0] for x,y in zip(spans,spans[1:])):
                    raise ValueError('invalid decoded spans')
