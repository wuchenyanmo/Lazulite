"""Explicit original/stem preparation and lossless, versioned prepared artifacts."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import numpy as np
from lazulite.models.audio import AudioBuffer, PreparedVocals, VocalAnalysis, VocalSegment, pcm_identity
from .activity import VocalConfig, analyze_vocals

SCHEMA = "lazulite-prepared-v1"
ANALYSIS_VERSION = "vocal-activity-preserved-v1"


def prepare_existing_vocals(vocals: AudioBuffer, *, original_id: str, original_duration: float,
                            reference: AudioBuffer | None = None, config: VocalConfig = VocalConfig(),
                            analysis: VocalAnalysis | None = None,
                            segments: tuple[VocalSegment, ...] | None = None,
                            analysis_identity: dict | None = None, separation: dict | None = None) -> PreparedVocals:
    if reference is not None and (reference.source_id != original_id or abs(reference.duration - original_duration) > 1 / reference.sample_rate):
        raise ValueError("reference does not match original")
    if analysis is not None and analysis_identity is None:
        raise ValueError("external analysis requires its explicit configuration and reference identity")
    identity = analysis_identity if analysis is not None else {
        "algorithm": ANALYSIS_VERSION, "config": asdict(config),
        "reference": "original_mix_mono_peak" if reference is not None else "full_scale",
        "reference_pcm": pcm_identity(reference) if reference is not None else None,
    }
    analysis = analysis if analysis is not None else analyze_vocals(vocals, config=config, reference=reference)
    # Own a read-only snapshot: caller mutation cannot invalidate a prepared artifact.
    samples = vocals.samples.copy()
    samples.setflags(write=False)
    snapshot = AudioBuffer(samples, vocals.sample_rate, vocals.source_id)
    return PreparedVocals(original_id, original_duration, snapshot, pcm_identity(snapshot), analysis,
                          analysis.segments if segments is None else tuple(segments),
                          "automatic" if segments is None else "manual", separation=separation or {"kind": "external_stem"},
                          analysis_identity=identity)


def prepare_vocals(original: AudioBuffer, *, separator, config: VocalConfig = VocalConfig()) -> PreparedVocals:
    if separator is None:
        raise ValueError("original preparation requires an explicit separator; use prepare_existing_vocals for stems")
    vocals = separator.separate(original)
    identity = getattr(separator, "cache_identity", None)
    identity = identity() if callable(identity) else identity
    return prepare_existing_vocals(vocals, original_id=original.source_id, original_duration=original.duration,
                                   reference=original, config=config,
                                   separation={"kind": "separated", "identity": identity})


def write_prepared(prepared: PreparedVocals, directory: str | Path) -> Path:
    # Immutable resource name, atomic manifest publication. No quantization/normalization.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared.__post_init__()
    name = prepared.pcm_id.removeprefix("sha256:") + ".npy"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, prepared.vocals.samples, allow_pickle=False)
    temporary.replace(directory / name)
    value = {"schema": SCHEMA, "original_id": prepared.original_id, "original_duration": prepared.original_duration,
             "origin": prepared.origin, "pcm_id": prepared.pcm_id, "resource": name,
             "sample_rate": prepared.vocals.sample_rate, "channels": prepared.vocals.samples.shape[0],
             "samples": prepared.vocals.samples.shape[1], "source_id": prepared.vocals.source_id,
             "analysis": asdict(prepared.analysis), "segments": [asdict(s) for s in prepared.segments],
             "segment_source": prepared.segment_source, "separation": prepared.separation,
             "analysis_identity": prepared.analysis_identity}
    with tempfile.NamedTemporaryFile("w", dir=directory, encoding="utf-8", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        temporary = Path(handle.name)
    target = directory / "manifest.json"
    temporary.replace(target)
    read_prepared(target)
    return target


def read_prepared(path: str | Path) -> PreparedVocals:
    path = Path(path)
    value = json.loads(path.read_text("utf-8"))
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported prepared schema")
    resource = (path.parent / value["resource"]).resolve()
    if resource.parent != path.parent.resolve():
        raise ValueError("prepared resource must be adjacent to manifest")
    samples = np.load(resource, allow_pickle=False)
    if samples.shape != (value["channels"], value["samples"]):
        raise ValueError("prepared layout mismatch")
    samples.setflags(write=False)
    audio = AudioBuffer(samples, value["sample_rate"], value["source_id"])
    analysis = dict(value["analysis"])
    analysis["segments"] = tuple(VocalSegment(**s) for s in analysis["segments"])
    analysis["diagnostics"] = tuple(analysis["diagnostics"])
    return PreparedVocals(value["original_id"], value["original_duration"], audio, value["pcm_id"],
                          VocalAnalysis(**analysis), tuple(VocalSegment(**s) for s in value["segments"]),
                          value["segment_source"], value["origin"], value["separation"], value["analysis_identity"])
