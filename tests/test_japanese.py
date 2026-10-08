import itertools
from dataclasses import replace
import numpy as np
import pytest
from lazulite.audio.concat import concatenate_cores
from lazulite.audio.preparation import prepare_existing_vocals
from lazulite.models.audio import AudioBuffer,VocalAnalysis,VocalSegment
from lazulite.models.japanese import JapaneseAlignmentRequest,JapaneseLine,JapaneseUnit
from lazulite.alignment.japanese.ctc import decode,NoFeasiblePath,SoftAnchor
from lazulite.alignment.japanese.backend import chunked_forward,NextFireConfig
from lazulite.alignment.japanese.run import align_japanese
from lazulite.alignment.japanese.profile import JaHepburnProfile
from lazulite.events import Cancelled


def test_mapping_union_seams_and_duration():
    audio=AudioBuffer(np.arange(96000,dtype=np.float32)[None],16000,'a')
    segments=tuple(VocalSegment(str(i),None,a,b,a,b,None,None) for i,(a,b) in enumerate([(1,2),(4,5),(1.5,2)]))
    c=concatenate_cores(audio,segments)
    np.testing.assert_array_equal(c.audio.samples[0],np.r_[audio.samples[0,16000:32000],audio.samples[0,64000:80000]])
    assert c.to_source(16000,endpoint='start')==64000
    assert c.to_source(16000,endpoint='end')==32000
    assert c.to_concat(48000) is None
    f=c.fragments(.9,1.1)
    assert [(x.start,x.end) for x in f]==[(1.9,2),(4,4.1)]
    assert sum(x.end-x.start for x in f)==pytest.approx(.2)
    assert concatenate_cores(audio,()).audio.samples.shape==(1,0)


def test_ctc_against_exhaustive_legal_paths_and_repeated_blank():
    rng=np.random.default_rng(43)
    for targets in ([1],[1,1],[1,2]):
        for _ in range(8):
            lp=rng.normal(size=(5,3))
            spans,score=decode(lp,targets,blank=0)
            legal=[]
            for path in itertools.product(range(3),repeat=5):
                collapsed=[x for i,x in enumerate(path) if x!=0 and (i==0 or path[i-1]!=x)]
                if collapsed==list(targets):
                    legal.append(sum(lp[t,x] for t,x in enumerate(path)))
            assert score==pytest.approx(max(legal))
    with pytest.raises(NoFeasiblePath):decode(np.zeros((2,3)),[1,1],blank=0)
    assert decode(np.zeros((3,3)),[1,1],blank=0)[0]==[(0,1),(2,3)]
    # Entry penalty happens once; staying on token does not pay again.
    spans,score=decode(np.array([[0.,10.]]*4),[1],blank=0,anchors=(SoftAnchor(0,0),))
    assert spans==[(0,4)] and score==40
    with pytest.raises(MemoryError):decode(np.zeros((3,3)),[1],blank=0,max_decode_memory_bytes=1)
    with pytest.raises(Cancelled):decode(np.zeros((3,3)),[1],blank=0,cancel=lambda:True)


def test_profile_and_chunk_coverage():
    p=JaHepburnProfile()
    assert p.unit_texts(['は','っ','ち','ー','ッ'],['ja']*5,[[]]*5)==['ha','t','chi','','']
    seen=[]
    y=np.linspace(-1,1,640480,dtype=np.float32)
    def forward(x):
        seen.append(x)
        return np.zeros(((len(x)-400)//320+1,32),np.float32)
    lp=chunked_forward(y,forward)
    assert lp.shape==((len(y)-400)//320+1,32)
    normalized=(y-y.mean())/np.sqrt(y.var()+1e-7)
    np.testing.assert_array_equal(seen[0],normalized[:352000])
    assert chunked_forward(np.zeros(399,np.float32),forward).shape==(0,32)
    with pytest.raises(ValueError):NextFireConfig('/none',revision='wrong')


def test_result_keeps_unknown_ids_manual_conflict_and_no_hidden_backend():
    audio=AudioBuffer(np.zeros((1,16000),np.float32),16000,'a')
    seg=VocalSegment('s',None,0,1,0,1,None,None)
    prepared=prepare_existing_vocals(audio,original_id='a',original_duration=1,
        analysis=VocalAnalysis('a',(seg,),(),False,0),analysis_identity={'fixture':True},segments=(seg,))
    request=JapaneseAlignmentRequest((JapaneseLine('l','あ漢',(JapaneseUnit('a',(0,1),'あ','ja','user'),JapaneseUnit('missing',(1,2),'','ja','unknown'))),))
    class Backend:
        def emissions(self,audio,**kwargs):
            result=np.full((49,32),-10.,np.float32)
            result[:,1]=0
            return result
    result=align_japanese(prepared,request,backend=Backend())
    assert result.status=='partial' and [u.unit_id for u in result.units]==['a','missing']
    assert result.execution['auto_conflict']
    automatic=replace(prepared,segment_source='automatic')
    result=align_japanese(automatic,request,backend=object())
    assert result.status=='unresolved' and result.units[0].issues[0].code=='no_vocal_evidence'
    with pytest.raises(ValueError,match='duplicate'):
        JapaneseAlignmentRequest((request.lines[0],request.lines[0]))
