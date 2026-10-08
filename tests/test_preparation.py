from dataclasses import replace
import json
import numpy as np
import pytest
from lazulite.audio.preparation import prepare_existing_vocals, prepare_vocals, read_prepared, write_prepared
from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment, pcm_identity


def fixture():
    audio = AudioBuffer(np.arange(32000, dtype=np.float32).reshape(2, -1) / 32000, 16000, 'original')
    segment = VocalSegment('s', None, 0, 1, .1, .9, None, None)
    analysis = VocalAnalysis('original', (segment,), (), False, 0)
    return audio, analysis


def test_prepared_lossless_round_trip_manual_evidence_and_owned_pcm(tmp_path):
    audio, analysis = fixture()
    prepared = prepare_existing_vocals(audio, original_id='original', original_duration=1,
        analysis=analysis, analysis_identity={'reference': 'none', 'algorithm': 'fixture'}, segments=analysis.segments)
    audio.samples[:] = 0
    result = read_prepared(write_prepared(prepared, tmp_path))
    assert result.pcm_id == prepared.pcm_id != pcm_identity(audio)
    np.testing.assert_array_equal(result.vocals.samples, prepared.vocals.samples)
    assert result.segment_source == 'manual' and result.analysis.is_vocal is False
    assert result.origin == 0 and not result.vocals.samples.flags.writeable
    with pytest.raises(ValueError, match='origin'):
        replace(result, origin=1)
    with pytest.raises(ValueError, match='duration'):
        replace(result, original_duration=2)
    resource = tmp_path / json.loads((tmp_path / 'manifest.json').read_text())['resource']
    np.save(resource, np.zeros((2,16000), np.float32))
    with pytest.raises(ValueError, match='identity'):
        read_prepared(tmp_path / 'manifest.json')


def test_prepare_separates_once_and_reuses_original_reference(monkeypatch):
    audio, analysis = fixture()
    calls = []
    class Separator:
        cache_identity = {'model': 'fixture'}
        def separate(self, value):
            calls.append(value)
            return value
    def analyze(value, *, config, reference):
        assert reference is audio
        return analysis
    monkeypatch.setattr(__import__('lazulite.audio.preparation', fromlist=['analyze_vocals']), 'analyze_vocals', analyze)
    prepared = prepare_vocals(audio, separator=Separator())
    assert len(calls) == 1 and prepared.separation['identity']['model'] == 'fixture'
    assert prepared.analysis_identity['reference_pcm'] == pcm_identity(audio)
    with pytest.raises(ValueError, match='explicit'):
        prepare_vocals(audio, separator=None)


def test_external_analysis_requires_provenance_and_rejects_bad_ranges():
    audio, analysis = fixture()
    with pytest.raises(ValueError, match='explicit'):
        prepare_existing_vocals(audio, original_id='original', original_duration=1, analysis=analysis)
    bad = replace(analysis, segments=(VocalSegment('s', None, 0, 2, 0, 2, None, None),))
    with pytest.raises(ValueError, match='outside'):
        prepare_existing_vocals(audio, original_id='original', original_duration=1,
            analysis=bad, analysis_identity={'reference':'none'})


def test_array_cache_rejects_missing_corrupt_and_unpublished_files(tmp_path):
    from lazulite.cache import FileCache
    cache=FileCache(tmp_path)
    np.save(tmp_path/'orphan.npy',np.ones(3))
    assert cache.get_array('orphan') is None
    cache.set_array('a',np.arange(6,dtype=np.float32))
    np.testing.assert_array_equal(cache.get_array('a'),np.arange(6,dtype=np.float32))
    path=tmp_path/cache.get_json('a')['resource'];path.write_bytes(b'broken')
    assert cache.get_array('a') is None
    path.unlink()
    assert cache.get_array('a') is None


def test_external_file_import_preserves_native_rate_channels_and_pcm(tmp_path):
    from scipy.io.wavfile import write
    from lazulite.audio.io import decode_native_audio
    samples=np.linspace(-.75,.75,22050,dtype=np.float32)
    p=tmp_path/'mono.wav';write(p,22050,samples)
    audio=decode_native_audio(p)
    assert audio.sample_rate==22050 and audio.samples.shape==(1,22050)
    np.testing.assert_array_equal(audio.samples[0],samples)
