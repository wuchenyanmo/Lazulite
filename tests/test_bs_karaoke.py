import numpy as np
import pytest
from lazulite.audio.bs_karaoke import BSKaraokeConfig, BSKaraokeSeparator
from lazulite.models.audio import AudioBuffer


@pytest.mark.parametrize('length', [8, 21, 31, 45])
def test_overlap_preserves_samples_and_identity(length):
    torch = pytest.importorskip('torch')
    separator = BSKaraokeSeparator(BSKaraokeConfig('unused'))
    separator._settings = {'audio':{'chunk_size':20}, 'inference':{'num_overlap':4}}
    separator._model = lambda x: x
    samples = np.random.default_rng(0).normal(size=(2,length)).astype('float32')
    output = separator.separate(AudioBuffer(samples,44100,'source'))
    assert output.source_id == 'source'
    assert output.sample_rate == 44100
    np.testing.assert_allclose(output.samples,samples,atol=3e-7)
    separator.close()
    assert separator._model is None
