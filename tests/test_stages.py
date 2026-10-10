from dataclasses import replace
import json
import numpy as np
import pytest
from fixtures.acoustic import Backend
from lazulite.alignment.japanese.run import compute_acoustics,decode_japanese,finalize_japanese,align_japanese,JapaneseConfig
from lazulite.alignment.japanese.stages import FrameMap
from lazulite.alignment.japanese.backend import chunked_forward,inspect_model
from lazulite.audio.preparation import prepare_existing_vocals
from lazulite.models.audio import AudioBuffer,VocalAnalysis,VocalSegment
from lazulite.models.japanese import JapaneseAlignmentRequest,JapaneseLine,JapaneseUnit
from lazulite.exchange import read_acoustic,write_acoustic,read_decode,write_decode


def inputs():
    audio=AudioBuffer(np.ones((1,24000),np.float32)*.1,8000,'voice')
    segments=(VocalSegment('a',None,0,1,0,1,None,None),VocalSegment('b',None,2,3,2,3,None,None))
    prepared=prepare_existing_vocals(audio,analysis=VocalAnalysis('voice',segments,is_vocal=True))
    request=JapaneseAlignmentRequest((JapaneseLine('l','あい',(JapaneseUnit('a',(0,1),'あ','ja','edited'),JapaneseUnit('i',(1,2),'い','ja','edited'))),))
    return prepared,request


@pytest.mark.parametrize('rate,stride,offset,columns,blank',[(16000,320,0,27,0),(8000,80,20,7,6),(22050,441,30,5,3)])
def test_dynamic_contract_and_all_stage_equivalence(tmp_path,rate,stride,offset,columns,blank):
    prepared,request=inputs()
    backend=Backend()
    backend.frames=FrameMap(rate,stride,stride+int(offset)+1,offset)
    backend.vocabulary_size=columns;backend.blank=blank;backend.vocab={'a':2,'i':1}
    acoustic=compute_acoustics(prepared,backend=backend,include_activity=True)
    cfg=JapaneseConfig(rest_blank=True,rest_token=True,hint_mode='off')
    whole=align_japanese(prepared,request,backend=backend,config=cfg)
    decoded=decode_japanese(acoustic,request,config=cfg)
    assert finalize_japanese(decoded)==whole
    assert align_japanese(None,request,acoustic=acoustic,config=cfg)==whole
    assert align_japanese(None,request,decoded=decoded,config=cfg)==whole
    assert backend.calls==2
    assert acoustic.mapping.to_source(rate,endpoint='start')==2*rate
    assert acoustic.mapping.to_source(rate,endpoint='end')==rate
    assert all(c['concat_start']>=offset/rate for u in whole.units for c in u.character_path)
    write_acoustic(tmp_path/'a.npz',acoustic)
    loaded=read_acoustic(tmp_path/'a.npz')
    assert finalize_japanese(decode_japanese(loaded,request,config=cfg))==whole
    write_decode(tmp_path/'d.json',decoded)
    assert finalize_japanese(read_decode(tmp_path/'d.json'))==whole
    with pytest.raises(ValueError,match='conflicts'): align_japanese(None,request,acoustic=acoustic,backend=backend)
    changed=replace(request,lines=(replace(request.lines[0],units=(replace(request.lines[0].units[0],reading='い'),request.lines[0].units[1])),))
    with pytest.raises(ValueError,match='targets'): align_japanese(None,changed,decoded=decoded,config=cfg)


def test_bad_explicit_acoustic_and_decoded_data_fail_without_model():
    prepared,request=inputs();a=compute_acoustics(prepared,backend=Backend())
    for fields in ({'blank_id':99},{'vocabulary':{'a':99}},{'log_probabilities':np.zeros((2,2,2))},
                   {'log_probabilities':np.ones_like(a.log_probabilities)},
                   {'frames':FrameMap(16000,320,400,999999)}, {'original_duration':.1},
                   {'rest':np.ones(1)}):
        with pytest.raises(ValueError): replace(a,**fields)
    with pytest.raises(ValueError,match='activity'): decode_japanese(a,request,config=JapaneseConfig(rest_token=True))
    d=decode_japanese(a,request)
    with pytest.raises(ValueError,match='spans'): replace(d,spans=((0,999999),)*2)
    with pytest.raises(ValueError,match='association'): replace(d,ranges={})
    from lazulite.events import Cancelled
    with pytest.raises(Cancelled): finalize_japanese(d,cancel=lambda:True)
    with pytest.raises(MemoryError): decode_japanese(a,request,config=JapaneseConfig(max_decode_memory_bytes=1))


@pytest.mark.parametrize('rate,stride,receptive',[(8000,80,100),(22050,147,350),(16000,320,400)])
def test_changed_stride_chunk_cores_have_exact_coverage(rate,stride,receptive):
    frames=FrameMap(rate,stride,receptive)
    y=np.arange(43*rate+113,dtype=np.float32)
    def forward(x):
        return np.repeat(x[np.arange(frames.count(len(x)))*stride,None],3,axis=1)
    out=chunked_forward(y,forward,frames=frames,vocabulary_size=3,normalize=False)
    np.testing.assert_array_equal(out[:,0],np.arange(frames.count(len(y)))*stride)


def test_lightweight_model_probe_accepts_modified_config_without_torch(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'wav2vec2','conv_kernel':[3,2],'conv_stride':[2,3],'pad_token_id':2,'vocab_size':5}))
    (tmp_path/'preprocessor_config.json').write_text(json.dumps({'feature_size':1,'sampling_rate':8000,'do_normalize':False}))
    (tmp_path/'vocab.json').write_text(json.dumps({'a':4,'i':0,'[PAD]':2}))
    (tmp_path/'model.safetensors').write_bytes(b'structure probe does not read model bytes')
    vocab,blank,columns,frames,normalize=inspect_model(tmp_path)
    assert (vocab['a'],blank,columns,frames.sample_rate,frames.stride_samples,frames.receptive_samples,normalize)==(4,2,5,8000,6,5,False)


def test_original_duration_allows_only_resampling_ceiling_sample():
    audio=AudioBuffer(np.ones((1,44101),np.float32)*.1,44100,'v')
    segment=VocalSegment('s',None,0,audio.duration,0,audio.duration,None,None)
    p=prepare_existing_vocals(audio,segments=(segment,))
    a=compute_acoustics(p,backend=Backend())
    assert a.mapping.pieces[-1].source_end/a.mapping.sample_rate > audio.duration
    assert a.mapping.pieces[-1].source_end/a.mapping.sample_rate-audio.duration < 1/a.mapping.sample_rate
    with pytest.raises(ValueError): replace(a,original_duration=audio.duration-.001)
