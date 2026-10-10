"""Model-independent heuristic evidence; these values are not probabilities."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import zlib
from lazulite.models.audio import VocalSegment
from lazulite.models.transcription import TranscriptionSegment

SUPPORTED_CONFIDENCE = {"generation_token_geometric_mean", "raw_token_geometric_mean"}


def _clip(value):
    return max(0.0, min(1.0, value))


def repetition(text: str) -> tuple[float, str]:
    clean = " ".join(text.split())
    best, shortened = 0.0, clean
    if len(clean) < 8:
        return best, shortened
    for split in range(
        max(3, len(clean) // 3), min(len(clean) - 3, 2 * len(clean) // 3 + 1)
    ):
        left, right = clean[:split].strip(), clean[split:].strip()
        if min(len(left), len(right)) < 3:
            continue
        ratio = min(len(left), len(right)) / max(len(left), len(right))
        if ratio < 0.70:
            continue
        score = SequenceMatcher(None, left, right).ratio() * ratio
        if score > best:
            best, shortened = score, left
    return _clip(best), shortened if best >= 0.82 else clean


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    repeat: float
    density: float | None
    risk: float
    components: tuple[tuple[str, float], ...]

    @property
    def usable(self):
        return self.repeat <= 0.60 and self.risk <= 0.55


def quality(text: str, segment: VocalSegment | None = None) -> QualityEvidence:
    clean = " ".join(text.split())
    repeat, _ = repetition(clean)
    raw = clean.encode("utf-8")
    compression = len(raw) / max(len(zlib.compress(raw)), 1)
    components = {"repetition": repeat, "compression": _clip((compression - 1.8) / 1.4)}
    density = None
    if segment is not None:
        density = _clip((len(clean) / max(segment.end - segment.start, 1e-6) - 9) / 10)
        components["density"] = density
        if segment.presence is not None:
            components["presence"] = (
                0.0
                if segment.presence >= 0.70
                else _clip(((0.70 - segment.presence) / 0.18) ** 2)
            )
        if segment.off_center_risk is not None:
            value = segment.off_center_risk
            components["off_center"] = (
                _clip(0.25 * value / 0.30)
                if value <= 0.30
                else _clip(0.25 + 0.75 * ((value - 0.30) / 0.25) ** 2)
            )
    top = sorted(components.values(), reverse=True)
    risk = sum(w * x for w, x in zip((0.50, 0.30, 0.20), top))
    return QualityEvidence(
        repeat, density, risk if clean else 1.0, tuple(components.items())
    )


def comparable_confidence(item: TranscriptionSegment) -> float | None:
    return item.confidence if item.confidence_kind in SUPPORTED_CONFIDENCE else None


def eligible_for_transcription(segment: VocalSegment) -> bool:
    return (
        segment.presence is None or segment.presence >= 0.20
    ) and segment.end - segment.start >= 0.80


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    policy: str = 'generic'

    def __post_init__(self):
        if self.policy not in {'generic','whisper-lexical-v1'}:
            raise ValueError('unknown evidence policy')


def transcription_quality(item, segment=None, config=EvidenceConfig()):
    """Shared DP/offset/history calculation. Missing summaries add no probability risk."""
    base=quality(item.text,segment)
    summary=item.probability
    if config.policy=='generic' or summary is None or summary.definition!='lexical_generation_probabilities-v1':
        return base
    components=dict(base.components)
    components.update({
        'mean_probability':_clip(1-summary.mean),
        'minimum_probability':_clip(((.45-summary.minimum)/.35)**1.6) if summary.minimum<.45 else 0.,
        'low_probability_fraction':_clip(((summary.low_fraction-.10)/.50)**1.4) if summary.low_fraction>.10 else 0.,
    })
    values=sorted(components.values(),reverse=True)
    return QualityEvidence(base.repeat,base.density,sum(w*v for w,v in zip((.5,.3,.2),values)),tuple(components.items()))
