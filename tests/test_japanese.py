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


def test_hint_calibration_full_request_coverage_zero_and_deleted_gap():
    from lazulite.models.japanese import StartHint,JapaneseUnitResult,Timing,Endpoint
    from lazulite.alignment.japanese.calibration import calibrate
    audio=AudioBuffer(np.zeros((1,160000),np.float32),16000,'a')
    c=concatenate_cores(audio,(VocalSegment('s',None,0,10,0,10,None,None),))
    lines=[];units=[];ranges={}
    for i in range(8):
        uid=str(i)
        lines.append(JapaneseLine(uid,'あ',(JapaneseUnit(uid,(0,1),'あ','ja','user'),),StartHint(float(i),'edited')))
        t=Timing(Endpoint(i+.2,'acoustic_path'),Endpoint(i+.5,'acoustic_path'))
        units.append(JapaneseUnitResult(uid,uid,'あ','a',t,t))
        ranges[uid]=(i,i+1)
    req=JapaneseAlignmentRequest(tuple(lines))
    anchors,r=calibrate(req,units,ranges,c)
    assert len(anchors)==8 and r['delta_seconds']==.2 and anchors[0].frame==10
    local=replace(req,lines=tuple(replace(l,start_hint=None) if i<2 or i>5 else l for i,l in enumerate(lines)))
    assert calibrate(local,units,ranges,c)[1]['reason']=='local_support'
    automatic=replace(req,lines=tuple(replace(l,start_hint=StartHint(float(i),'automatic','japanese')) for i,l in enumerate(lines)))
    assert calibrate(automatic,units,ranges,c)[1]['reason']=='no_eligible_hints'
    # Calibration does not snap corrected hints in deleted gaps.
    c=concatenate_cores(audio,(VocalSegment('a',None,0,.1,0,.1,None,None),VocalSegment('b',None,2,10,2,10,None,None)))
    assert calibrate(req,units,ranges,c)[1]['reason']=='insufficient_mapped_coverage'


def test_completion_interpolates_only_same_piece_and_keeps_raw_cross_gap():
    from lazulite.models.japanese import JapaneseUnitResult,Timing,Endpoint,Fragment
    from lazulite.alignment.japanese.completion import complete_units
    audio=AudioBuffer(np.zeros((1,160000),np.float32),16000,'a')
    concat=concatenate_cores(audio,tuple(VocalSegment(str(i),None,a,b,a,b,None,None) for i,(a,b) in enumerate([(0,4),(8,10)])))
    req=JapaneseAlignmentRequest((JapaneseLine('l','あいう',tuple(JapaneseUnit(str(i),(i,i+1),'あ','ja','user') for i in range(3))),))
    def unit(i,a=None,b=None,**kw):
        t=Timing(Endpoint(a,'acoustic_path') if a is not None else None,Endpoint(b,'acoustic_path') if b is not None else None)
        return JapaneseUnitResult(str(i),'l','あ','a',t,t,**kw)
    us=(unit(0,0,1),unit(1),unit(2,3,4))
    out=complete_units(req,us,concat,10)
    assert out[0]==us[0] and out[2]==us[2] and out[1].raw_timing==Timing()
    assert (out[1].proposed_timing.start.time,out[1].proposed_timing.end.time)==(1,3)
    assert out[1].proposed_timing.start.source=='interpolated'
    us=(unit(0,0,1),unit(1),unit(2,8,9))
    assert complete_units(req,us,concat,10)[1].proposed_timing==Timing()
    cross=replace(unit(1,2,9),proposed_timing=Timing(),fragments=(Fragment(2,4,0),Fragment(8,9,1)),crosses_removed_gap=True)
    us=(unit(0,0,1),cross,unit(2,9,10))
    assert complete_units(req,us,concat,10)[1].proposed_timing==Timing()
    us=(unit(0,8,8.2),cross,unit(2,9,10))
    out=complete_units(req,us,concat,10)
    assert out[1].proposed_timing.start.time==8.2 and out[1].raw_timing==cross.raw_timing


def test_reading_change_reuses_acoustics_and_core_change_invalidates(tmp_path):
    from lazulite.cache import FileCache
    audio=AudioBuffer(np.zeros((1,32000),np.float32),16000,'a')
    seg=VocalSegment('s',None,0,2,0,2,None,None)
    prepared=prepare_existing_vocals(audio,original_id='a',original_duration=2,
        analysis=VocalAnalysis('a',(seg,),(),True,2),analysis_identity={'reference':'none'})
    req=JapaneseAlignmentRequest((JapaneseLine('l','あ',(JapaneseUnit('u',(0,1),'あ','ja','user'),)),))
    class Backend:
        cache_identity={'fixture':'trusted-content-1'}
        calls=0
        def emissions(self,audio,**kw):
            self.calls+=1
            return np.zeros(((audio.samples.shape[1]-400)//320+1,32),np.float32)
    b=Backend();cache=FileCache(tmp_path)
    first=align_japanese(prepared,req,backend=b,cache=cache)
    second=align_japanese(prepared,replace(req,lines=(replace(req.lines[0],units=(replace(req.lines[0].units[0],reading='い'),)),)),backend=b,cache=cache)
    assert b.calls==1 and first.execution['acoustic_id']==second.execution['acoustic_id']
    manual=replace(prepared,segments=(replace(seg,core_start=1),),segment_source='manual')
    third=align_japanese(manual,req,backend=b,cache=cache)
    assert b.calls==2 and third.execution['acoustic_id']!=first.execution['acoustic_id']


def test_interpolation_does_not_hide_unresolved_reading_status():
    from lazulite.alignment.japanese.run import _result
    from lazulite.models.japanese import JapaneseUnitResult,JapaneseIssue,Timing,Endpoint
    request=JapaneseAlignmentRequest((JapaneseLine('l','漢',(JapaneseUnit('u',(0,1),'','ja','unknown'),)),))
    unit=JapaneseUnitResult('u','l','','',Timing(),Timing(Endpoint(1,'interpolated'),Endpoint(2,'interpolated')))
    result=_result(request,(unit,),{}, {},(JapaneseIssue('request','partial_targets'),))
    assert result.status=='partial' and result.coverage['proposed_complete']==1 and result.coverage['raw_complete']==0
