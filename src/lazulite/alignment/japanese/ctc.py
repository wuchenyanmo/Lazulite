"""Float64 global CTC, rolling scores and uint8 backpointers.

Recurrence/tie order follows MiliKara ff2d2f54706ecb140bc5b275037eff64187ef6ef
(MIT; see THIRD_PARTY_LICENSES). No free tail or blank merging; optional per-frame soft activity costs.
"""
from dataclasses import dataclass
import numpy as np
from lazulite.events import checkpoint

VERSION = 'global-ctc-f64-v2'
DEFAULT_MEMORY = 512 * 1024 * 1024


class NoFeasiblePath(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SoftAnchor:
    token_index: int
    frame: float
    sigma_frames: float = 20
    weight: float = 4


def decode(logp, targets, blank, *, anchors=(), max_decode_memory_bytes=DEFAULT_MEMORY,
           cancel=None, event_sink=None, token_lines=None, frame_seconds=.02,
           line_blank=False, rest_blank=False, rest_token=False, rest=None):
    lp = np.asarray(logp)
    targets = np.asarray(targets, dtype=np.int64)
    if lp.ndim != 2 or np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError('invalid log probabilities')
    T, V = lp.shape
    if not 0 <= blank < V or np.any(targets < 0) or np.any(targets >= V) or np.any(targets == blank):
        raise ValueError('invalid CTC targets/blank')
    if T == 0 or not len(targets):
        raise NoFeasiblePath('empty frames or targets')
    S = 2 * len(targets) + 1
    # Includes backpointers, score/work vectors, and returned frame path; lp stays in caller dtype.
    required = T*S + S*128 + T*17
    if required > max_decode_memory_bytes:
        raise MemoryError(f'CTC requires {required} bytes, budget {max_decode_memory_bytes}')
    if T < len(targets) + np.count_nonzero(targets[1:] == targets[:-1]):
        raise NoFeasiblePath('insufficient frames for repeated targets')
    z = np.full(S, blank, np.int64)
    z[1::2] = targets
    skip = np.zeros(S, bool)
    skip[2:] = (z[2:] != blank) & (z[2:] != z[:-2])
    if not np.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError('invalid frame duration')
    gap_states = np.zeros(S, bool)
    if line_blank or rest_blank:
        if token_lines is None or len(token_lines) != len(targets):
            raise ValueError('token line identities required for blank costs')
        gap_states[2:-1:2] = np.asarray(token_lines[1:]) == np.asarray(token_lines[:-1])
    if rest_blank or rest_token:
        rest = np.asarray(rest, dtype=np.float64)
        if rest.shape != (T,) or not np.isfinite(rest).all() or np.any((rest < 0) | (rest > 1)):
            raise ValueError('rest must be one finite 0–1 value per frame')
    def frame_cost(t):
        c = np.zeros(S, np.float64)
        c[gap_states] = -frame_seconds * (int(line_blank) + (4 * rest[t] if rest_blank else 0))
        if rest_token:
            c[1::2] = -25 * frame_seconds * rest[t]
        return c
    use_priors = line_blank or rest_blank or rest_token
    anchor_states = {}
    for a in anchors:
        if not 0 <= a.token_index < len(targets) or a.sigma_frames <= 0 or a.weight < 0 or not np.isfinite([a.frame,a.sigma_frames,a.weight]).all():
            raise ValueError('invalid soft anchor')
        state = 2*a.token_index+1
        if state in anchor_states:
            raise ValueError('duplicate anchor token')
        anchor_states[state] = a
    def costs(t):
        c = np.zeros(S, np.float64)
        for state,a in anchor_states.items():
            d = abs((t-a.frame)/a.sigma_frames)
            c[state] = -a.weight*(.5*d*d if d <= 1 else d-.5)
        return c
    checkpoint(cancel, event_sink, 'decode', 0,T)
    bp = np.zeros((T,S),np.uint8)
    prev = np.full(S,-np.inf,np.float64)
    prev[0] = lp[0,blank]
    prev[1] = float(lp[0,z[1]])+costs(0)[1]
    if use_priors: prev += frame_cost(0)
    for t in range(1,T):
        if t % 128 == 0:
            checkpoint(cancel,event_sink,'decode',t,T)
        c = costs(t)
        best = prev.copy()
        one = np.r_[-np.inf,prev[:-1]] + c
        take = one > best
        best[take] = one[take]; bp[t,take] = 1
        two = np.r_[-np.inf,-np.inf,prev[:-2]] + c
        take = skip & (two > best)
        best[take] = two[take]; bp[t,take] = 2
        prev = best + lp[t,z].astype(np.float64)
        if use_priors: prev += frame_cost(t)
    state = S-1 if prev[-1] >= prev[-2] else S-2
    if not np.isfinite(prev[state]):
        raise NoFeasiblePath('no feasible global path')
    score = float(prev[state])
    path = np.empty(T,np.int64)
    for t in range(T-1,-1,-1):
        path[t] = state
        state -= int(bp[t,state])
    checkpoint(cancel,event_sink,'decode',T,T)
    spans = []
    for i in range(len(targets)):
        frames = np.flatnonzero(path == 2*i+1)
        spans.append((int(frames[0]),int(frames[-1])+1))
    return spans, score
