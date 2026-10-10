from dataclasses import replace
import numpy as np
import pytest
from lazulite.audio.preparation import prepare_existing_vocals, prepare_vocals
from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment


def fixture():
    audio = AudioBuffer(np.arange(32000, dtype=np.float32).reshape(2,-1)/32000,16000,'original')
    segment = VocalSegment('s',None,0,1,.1,.9,None,None)
    return audio,VocalAnalysis('original',(segment,),(),False,0)


def test_direct_complete_vocals_preserve_manual_selection_and_optional_description():
    audio,analysis=fixture()
    result=prepare_existing_vocals(audio,analysis=analysis,segments=analysis.segments)
    assert result.vocals is audio and result.original_duration==1
    assert result.segment_source=='manual' and result.analysis.is_vocal is False
    assert result.analysis_description=={'kind':'external'}
    with pytest.raises(ValueError,match='origin'): replace(result,origin=1)
    with pytest.raises(ValueError,match='duration'): replace(result,original_duration=2)


def test_prepare_separates_each_run_and_uses_original_reference(monkeypatch):
    audio,analysis=fixture();calls=[]
    class Separator:
        def separate(self,value): calls.append(value);return value
    def analyze(value,*,config,reference):
        assert reference is audio
        return analysis
    monkeypatch.setattr(__import__('lazulite.audio.preparation',fromlist=['analyze_vocals']),'analyze_vocals',analyze)
    for _ in range(2):
        result=prepare_vocals(audio,separator=Separator())
        assert result.analysis_description['reference']=='original_mix_mono_peak'
    assert len(calls)==2
    with pytest.raises(ValueError,match='explicit'): prepare_vocals(audio,separator=None)


def test_analysis_association_and_ranges_are_checked():
    audio,analysis=fixture()
    bad=replace(analysis,segments=(VocalSegment('s',None,0,2,0,2,None,None),))
    with pytest.raises(ValueError,match='outside'): prepare_existing_vocals(audio,analysis=bad)
    with pytest.raises(ValueError,match='identities'): prepare_existing_vocals(audio,analysis=replace(analysis,source_id='other'))


def test_external_file_import_preserves_native_rate_channels_and_pcm(tmp_path):
    from scipy.io.wavfile import write
    from lazulite.audio.io import decode_native_audio
    samples=np.linspace(-.75,.75,22050,dtype=np.float32)
    p=tmp_path/'mono.wav';write(p,22050,samples)
    audio=decode_native_audio(p)
    assert audio.sample_rate==22050 and audio.samples.shape==(1,22050)
    np.testing.assert_array_equal(audio.samples[0],samples)
