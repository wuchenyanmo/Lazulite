"""Float64 global CTC, rolling scores and uint8 backpointers.

Recurrence/tie order follows MiliKara ff2d2f54706ecb140bc5b275037eff64187ef6ef
(MIT; see THIRD_PARTY_LICENSES). No free tail or blank merging; optional per-frame soft activity costs.
"""
from dataclasses import dataclass
import numpy as np
from lazulite.events import checkpoint

VERSION = 'global-ctc-f64-v3'
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
           cancel=None, event_sink=None, token_lines=None, frame_seconds=None,
           line_blank=False, rest_blank=False, rest_token=False, rest=None, seam_cost=None):
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
    if frame_seconds is None and not (line_blank or rest_blank or rest_token):
        frame_seconds = 1.0  # No duration-dependent term is evaluated.
    if frame_seconds is None or not np.isfinite(frame_seconds) or frame_seconds <= 0:
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
    owners = np.full(S, -1, dtype=np.int64)
    if seam_cost is not None:
        seam_cost = np.asarray(seam_cost, dtype=np.float64)
        if seam_cost.shape != (T,) or not np.isfinite(seam_cost).all() or np.any(seam_cost < 0):
            raise ValueError('invalid seam costs')
        if token_lines is None or len(token_lines) != len(targets):
            raise ValueError('token line identities required for seam costs')
        identities = {line:i for i,line in enumerate(dict.fromkeys(token_lines))}
        owners[1::2] = [identities[line] for line in token_lines]
        owners[2:-1:2] = np.where(owners[1:-2:2] == owners[3::2], owners[3::2], -1)
    stay_owned = owners >= 0
    shifted_one = np.empty(S, np.int64); shifted_one[0] = -2; shifted_one[1:] = owners[:-1]
    shifted_two = np.empty(S, np.int64); shifted_two[:2] = -2; shifted_two[2:] = owners[:-2]
    one_owned = stay_owned & (owners == shifted_one)
    two_owned = stay_owned & (owners == shifted_two)
    use_priors = line_blank or rest_blank or rest_token
    anchor_states = {}
    for a in anchors:
        if not 0 <= a.token_index < len(targets) or a.sigma_frames <= 0 or a.weight < 0 or not np.isfinite([a.frame,a.sigma_frames,a.weight]).all():
            raise ValueError('invalid soft anchor')
        state = 2*a.token_index+1
        if state in anchor_states:
            raise ValueError('duplicate anchor token')
        anchor_states[state] = a
    c = np.zeros(S, np.float64)
    fc = np.zeros(S, np.float64)
    best = np.empty(S, np.float64)
    cand = np.empty(S, np.float64)
    take = np.empty(S, bool)
    one_shifted = np.empty(S, np.float64); one_shifted[0] = -np.inf
    two_shifted = np.empty(S, np.float64); two_shifted[:2] = -np.inf
    gap_const = -frame_seconds * int(line_blank)
    def update_anchors(t):
        for state, a in anchor_states.items():
            d = abs((t - a.frame) / a.sigma_frames)
            c[state] = -a.weight * (.5 * d * d if d <= 1 else d - .5)
    checkpoint(cancel, event_sink, 'decode', 0,T)
    bp = np.zeros((T,S),np.uint8)
    prev = np.full(S,-np.inf,np.float64)
    prev[0] = lp[0,blank]
    if anchor_states:
        update_anchors(0)
        prev[1] = float(lp[0,z[1]]) + c[1]
    else:
        prev[1] = float(lp[0,z[1]])
    if use_priors:
        if line_blank or rest_blank:
            fc[gap_states] = gap_const - (4 * frame_seconds * rest[0] if rest_blank else 0.0)
        if rest_token:
            fc[1::2] = -25 * frame_seconds * rest[0]
        prev += fc
    for t in range(1,T):
        if t % 128 == 0:
            checkpoint(cancel,event_sink,'decode',t,T)
        if anchor_states:
            update_anchors(t)
        penalty = 0.0 if seam_cost is None else seam_cost[t]
        np.copyto(best, prev)
        if penalty:
            best[stay_owned] -= penalty
        one_shifted[1:] = prev[:-1]
        if anchor_states:
            np.add(one_shifted, c, out=cand)
        else:
            np.copyto(cand, one_shifted)
        if penalty:
            cand[one_owned] -= penalty
        np.greater(cand, best, out=take)
        best[take] = cand[take]; bp[t,take] = 1
        two_shifted[2:] = prev[:-2]
        if anchor_states:
            np.add(two_shifted, c, out=cand)
        else:
            np.copyto(cand, two_shifted)
        if penalty:
            cand[two_owned] -= penalty
        np.greater(cand, best, out=take)
        take &= skip
        best[take] = cand[take]; bp[t,take] = 2
        np.add(best, lp[t,z], out=prev)
        if use_priors:
            if rest_blank:
                fc[gap_states] = gap_const - 4 * frame_seconds * rest[t]
            if rest_token:
                fc[1::2] = -25 * frame_seconds * rest[t]
            prev += fc
    state = S-1 if prev[-1] >= prev[-2] else S-2
    if not np.isfinite(prev[state]):
        raise NoFeasiblePath('no feasible global path')
    score = float(prev[state])
    path = np.empty(T,np.int64)
    for t in range(T-1,-1,-1):
        path[t] = state
        state -= int(bp[t,state])
    checkpoint(cancel,event_sink,'decode',T,T)
    target_states = np.arange(1, S, 2)
    starts = np.searchsorted(path, target_states, side='left')
    ends = np.searchsorted(path, target_states, side='right')
    return [(int(a), int(b)) for a, b in zip(starts, ends)], score
