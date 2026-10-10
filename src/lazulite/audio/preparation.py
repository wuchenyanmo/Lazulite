"""In-memory complete vocals and explicitly selected analysis/segments."""
from dataclasses import asdict
from lazulite.models.audio import AudioBuffer, PreparedVocals, VocalAnalysis, VocalSegment
from .activity import VocalConfig, analyze_vocals

ANALYSIS_VERSION = "vocal-activity-preserved-v1"


def prepare_existing_vocals(vocals: AudioBuffer, *, original_id: str | None = None,
                            original_duration: float | None = None,
                            reference: AudioBuffer | None = None, config: VocalConfig = VocalConfig(),
                            analysis: VocalAnalysis | None = None,
                            segments: tuple[VocalSegment, ...] | None = None,
                            analysis_description: dict | None = None, separation: dict | None = None) -> PreparedVocals:
    duration = vocals.duration if original_duration is None else original_duration
    if reference is not None and abs(reference.duration - duration) > 1 / reference.sample_rate:
        raise ValueError("reference does not match complete duration")
    external = analysis is not None
    analysis = analysis if external else analyze_vocals(vocals, config=config, reference=reference)
    description = (analysis_description or {"kind": "external"}) if external else {
        "algorithm": ANALYSIS_VERSION, "config": asdict(config),
        "reference": "original_mix_mono_peak" if reference is not None else "full_scale",
    }
    return PreparedVocals(original_id or vocals.source_id, duration, vocals, analysis,
                          analysis.segments if segments is None else tuple(segments),
                          "automatic" if segments is None else "manual",
                          separation=separation or {"kind": "external_stem"}, analysis_description=description)


def prepare_vocals(original: AudioBuffer, *, separator, config: VocalConfig = VocalConfig(),
                   cancel=None, event_sink=None) -> PreparedVocals:
    from lazulite.events import checkpoint
    if separator is None:
        raise ValueError("original preparation requires an explicit separator")
    checkpoint(cancel,event_sink,'prepare')
    vocals = separator.separate(original)
    checkpoint(cancel,event_sink,'analysis')
    prepared = prepare_existing_vocals(vocals, original_id=original.source_id,
        original_duration=original.duration, reference=original, config=config,
        separation={"kind": "separated"})
    checkpoint(cancel)
    return prepared
