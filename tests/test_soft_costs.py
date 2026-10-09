import itertools
from dataclasses import replace
import numpy as np
import pytest
from lazulite.alignment.japanese.ctc import decode
from lazulite.alignment.japanese.activity import original_rest, map_rest
from lazulite.alignment.japanese.run import JapaneseConfig, align_japanese
from lazulite.audio.concat import concatenate_cores
from lazulite.audio.preparation import prepare_existing_vocals
from lazulite.models.audio import AudioBuffer, VocalAnalysis, VocalSegment
from lazulite.models.japanese import JapaneseAlignmentRequest, JapaneseLine, JapaneseUnit
from lazulite.cache import FileCache


@pytest.mark.parametrize('switches', list(itertools.product([False, True], repeat=3)))
def test_costs_against_exhaustive_state_paths(switches):
    rng = np.random.default_rng(56)
    for targets in ([1, 1], [1, 2]):
        for lines in (['a', 'a'], ['a', 'b']):
            lp = rng.normal(size=(6, 3))
            rest = np.linspace(0, 1, 6)
            lb, rb, rt = switches
            spans, score = decode(lp, targets, blank=0, token_lines=lines, rest=rest,
                line_blank=lb, rest_blank=rb, rest_token=rt)
            z = [0, targets[0], 0, targets[1], 0]
            scores = []
            for path in itertools.product(range(5), repeat=6):
                if path[0] not in (0, 1) or path[-1] not in (3, 4): continue
                if any(not (b == a or b == a+1 or b == a+2 and z[b] != 0 and z[b] != z[a]) for a,b in zip(path,path[1:])): continue
                scores.append(sum(lp[t,z[s]] - (.02*(lb+4*rb*rest[t]) if s==2 and lines[0]==lines[1] else 0)
                    - (.02*25*rt*rest[t] if s%2 else 0) for t,s in enumerate(path)))
            assert score == pytest.approx(max(scores))
            if not any(switches): assert (spans,score) == decode(lp,targets,blank=0)


def test_original_activity_and_seam_mapping():
    values = np.r_[np.ones(16000)*.2, np.zeros(32000), np.ones(16000)*.02].astype(np.float32)
    audio = AudioBuffer(values[None],16000,'voice')
    rest, ref = original_rest(audio)
    assert ref is not None and rest[50] == 0 and rest[200] == 1
    assert len(rest)==400
    segs = tuple(VocalSegment(str(i),None,a,b,a,b,None,None) for i,(a,b) in enumerate([(0,.99),(3,3.51)]))
    concat = concatenate_cores(audio,segs)
    # Known original envelope: seam frame contains 10 ms from each distant piece.
    synthetic = np.r_[np.zeros(300),np.ones(100)].astype(np.float32)
    mapped = map_rest(synthetic,concat,74)
    assert mapped[48]==0 and mapped[49]==.5 and mapped[50]==1
    # Smoothing was done on original audio; the rest at the second piece start
    # still sees its actual preceding silence, not the loud first piece.
    actual = map_rest(rest,concat,74)
    assert actual[50] > .8
    assert len(original_rest(AudioBuffer(np.zeros((1,80),np.float32),16000,'short'))[0])==1


def test_combinations_cache_isolation_and_acoustic_reuse(tmp_path, monkeypatch):
    import lazulite.alignment.japanese.activity as activity
    import lazulite.alignment.japanese.run as run
    from lazulite.alignment.japanese.ctc import SoftAnchor
    from lazulite.alignment.japanese import calibration
    audio=AudioBuffer(np.ones((1,32000),np.float32)*.1,16000,'a')
    seg=VocalSegment('s',None,0,2,0,2,None,None)
    prepared=prepare_existing_vocals(audio,original_id='a',original_duration=2,
        analysis=VocalAnalysis('a',(seg,),(),True,2),analysis_identity={'fixture':True})
    req=JapaneseAlignmentRequest((JapaneseLine('l','あい',(JapaneseUnit('u',(0,1),'あ','ja','user'),JapaneseUnit('v',(1,2),'い','ja','user'))),))
    class Backend:
        cache_identity={'fixture':'soft-costs'}
        calls=0
        def emissions(self,audio,**kw):
            self.calls+=1
            return np.zeros(((audio.samples.shape[1]-400)//320+1,32),np.float32)
    b=Backend();cache=FileCache(tmp_path);ids=set();activity_calls=[];decodes=[]
    original=activity.frame_activity
    def tracked(*args): activity_calls.append(True);return original(*args)
    monkeypatch.setattr(activity,'frame_activity',tracked)
    original_decode=run.decode
    def capture(*args,**kwargs): decodes.append(kwargs);return original_decode(*args,**kwargs)
    monkeypatch.setattr(run,'decode',capture)
    monkeypatch.setattr(calibration,'calibrate',lambda *args: ((SoftAnchor(0,0),),{}))
    for flags in itertools.product([False,True],repeat=3):
        config=JapaneseConfig(line_blank=flags[0],rest_blank=flags[1],rest_token=flags[2])
        before=len(activity_calls)
        result=align_japanese(prepared,req,backend=b,config=config,cache=cache)
        assert len(activity_calls)-before == int(flags[1] or flags[2])
        assert result.calibration['adopted']=='second_pass'
        assert [(x['line_blank'],x['rest_blank'],x['rest_token']) for x in decodes[-2:]]==[flags,flags]
        ids.add(result.execution['decode_id'])
        repeated=align_japanese(prepared,req,backend=b,config=config,cache=cache)
        assert repeated.units==result.units
    assert len(ids)==8 and b.calls==1
    with pytest.raises(ValueError): JapaneseConfig(line_blank=1)


def test_disabled_costs_match_frozen_v1(tmp_path):
    # Frozen recurrence extracted from the previous release, independent of v2.
    import subprocess, importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','show','8aef75f:src/lazulite/alignment/japanese/ctc.py'],cwd=root,text=True)
    path=tmp_path/'ctc_v1.py';path.write_text(source)
    import sys
    spec=importlib.util.spec_from_file_location('ctc_v1',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    rng=np.random.default_rng(37)
    for targets in ([1,2,1],[1,1,2],[1]):
        for _ in range(20):
            lp=rng.normal(size=(12,4)).astype(np.float32)
            assert decode(lp,targets,blank=0)==old.decode(lp,targets,blank=0)
