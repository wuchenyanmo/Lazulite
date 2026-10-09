"""Independent Japanese fixed baseline, retaining every requested object ID."""
from dataclasses import asdict, dataclass, replace
import numpy as np
from lazulite.audio.concat import concatenate_cores
from lazulite.cache import cache_key
from lazulite.events import checkpoint, ProgressEvent
from lazulite.models.audio import PreparedVocals
from lazulite.models.japanese import (
    Endpoint, Timing, JapaneseAlignmentRequest, JapaneseAlignmentResult, JapaneseUnitResult, JapaneseIssue,
)
from .backend import VOCAB, VERSION as ACOUSTIC_VERSION
from .ctc import decode, NoFeasiblePath, DEFAULT_MEMORY, VERSION as CTC_VERSION
from .profile import JaHepburnProfile, is_hold
from .kana import is_kana_text


@dataclass(frozen=True, slots=True)
class JapaneseConfig:
    hint_mode: str = 'auto'
    max_decode_memory_bytes: int = DEFAULT_MEMORY
    completion: bool = True
    line_blank: bool = False
    rest_blank: bool = False
    rest_token: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.line_blank, self.rest_blank, self.rest_token)):
            raise ValueError('soft cost switches must be boolean')
        if self.hint_mode not in {'off','auto'} or self.max_decode_memory_bytes <= 0:
            raise ValueError('invalid Japanese configuration')


def _profile(request):
    texts, tokens, ranges, problems = {}, [], {}, {}
    profile = JaHepburnProfile()
    for line in request.lines:
        # Explicit language contract: foreign kana does not silently become Japanese.
        valid = [u.reading if u.reading_language == 'ja' and (is_kana_text(u.reading) or is_hold(u.reading))
                 or u.reading_language in {'en','latin'} and all(ord(c)<128 for c in u.reading) else '' for u in line.units]
        values = profile.unit_texts(valid, ['ja' if u.reading_language=='ja' else 'en' for u in line.units], [[] for u in line.units])
        for u,text in zip(line.units,values):
            texts[u.unit_id] = text
            begin = len(tokens)
            if text and all(c in VOCAB and c not in '| ' for c in text):
                tokens.extend(VOCAB[c] for c in text)
            else:
                problems[u.unit_id] = 'profile_hold' if u.reading_language=='ja' and is_hold(u.reading) else 'unresolved_reading'
            ranges[u.unit_id] = (begin,len(tokens))
    return texts,tokens,ranges,problems


def _intervals(request, spans, ranges):
    raw = {}
    sources = {}
    for line in request.lines:
        rows = []
        for u in line.units:
            a,b = ranges[u.unit_id]
            # FrameMap(16000,320,0,0), integer milliseconds, matching frozen upstream.
            rows.append([spans[a][0]*20,spans[b-1][1]*20] if b>a else None)
        i=0
        while i<len(rows):
            j=i+1
            while j<len(rows) and is_hold(line.units[j].reading) and line.units[j].reading_language=='ja' and rows[j] is None:
                j+=1
            if j>i+1 and rows[i] is not None and not is_hold(line.units[i].reading):
                a,b=rows[i]
                nxt=next((r[0] for r in rows[j:] if r is not None),None)
                end=b+(min(800,max(0,nxt-b)) if nxt is not None else 300)
                end=max(end,a+20*(j-i))
                if nxt is not None and nxt>=a+20*(j-i):
                    end=min(end,nxt)
                step=(end-a)/(j-i)
                for k in range(i,j):
                    rows[k]=[int(round(a+step*(k-i))),int(round(a+step*(k-i+1))) if k<j-1 else int(end)]
                    sources[line.units[k].unit_id]='profile_hold'
            i=max(j,i+1)
        for u,row in zip(line.units,rows):
            raw[u.unit_id] = None if row is None else (row[0]/1000,row[1]/1000)
    return raw,sources


def _units(request,texts,ranges,problems,spans,concat):
    intervals,sources=_intervals(request,spans,ranges)
    units=[]
    for line in request.lines:
        for u in line.units:
            interval=intervals[u.unit_id]
            issue=[]
            if u.unit_id in problems:
                issue.append(JapaneseIssue(u.unit_id,problems[u.unit_id]))
            fragments=()
            timing=Timing()
            if interval is not None:
                a,b=interval
                # Preserve baseline hold extension as evidence; report clipped acoustic domain.
                if b>concat.audio.duration:
                    issue.append(JapaneseIssue(u.unit_id,'profile_outside_audio',{'end':b}))
                    b=concat.audio.duration
                if b>a:
                    fragments=concat.fragments(a,b)
                    source=sources.get(u.unit_id,'acoustic_path')
                    timing=Timing(Endpoint(fragments[0].start,source,(u.unit_id,)), Endpoint(fragments[-1].end,source,(u.unit_id,)))
            cross=len(fragments)>1
            if cross:
                issue.append(JapaneseIssue(u.unit_id,'crosses_removed_gap'))
            a,b=ranges[u.unit_id]
            chars=tuple({'character':texts[u.unit_id][k-a], 'frames':s,
                         'concat_start':s[0]*.02,'concat_end':s[1]*.02} for k,s in enumerate(spans[a:b],a))
            units.append(JapaneseUnitResult(u.unit_id,line.line_id,u.reading,texts[u.unit_id],timing,
                Timing() if cross else timing,fragments,interval,cross,
                sum(y.start-x.end for x,y in zip(fragments,fragments[1:])),chars,tuple(issue)))
    return tuple(units)


def _result(request,units,execution,calibration,issues=()):
    def complete(t): return t.start is not None and t.end is not None and t.start.time < t.end.time
    raw=sum(complete(u.raw_timing) for u in units)
    full=sum(complete(u.proposed_timing) for u in units)
    single=sum((u.proposed_timing.start is None)!=(u.proposed_timing.end is None) for u in units)
    derived=sum(any(e is not None and e.source!='acoustic_path' for e in (u.proposed_timing.start,u.proposed_timing.end)) for u in units)
    lines=[]
    for line in request.lines:
        group=[u for u in units if u.line_id==line.line_id]
        starts=[u.proposed_timing.start.time for u in group if u.proposed_timing.start is not None]
        ends=[u.proposed_timing.end.time for u in group if u.proposed_timing.end is not None]
        lines.append({'line_id':line.line_id,'start':min(starts) if starts else None,'end':max(ends) if ends else None,
                      'first_unit_complete': bool(group) and complete(group[0].proposed_timing)})
    status='complete' if full==len(units) and units and not any(i.code=='partial_targets' for i in issues) else 'partial' if full or single else 'unresolved'
    return JapaneseAlignmentResult(units,status,tuple(lines),{'total':len(units),'raw_complete':raw,'proposed_complete':full,
        'single_endpoint':single,'unresolved':len(units)-full-single,'derived':derived,
        'profile_derived':sum(any(e is not None and e.source=='profile_hold' for e in (u.raw_timing.start,u.raw_timing.end)) for u in units),
        'completion_derived':sum(u.proposed_timing != u.raw_timing and any(e is not None for e in (u.proposed_timing.start,u.proposed_timing.end)) for u in units),
        'empty_targets':sum(not u.model_text for u in units)},calibration,execution,tuple(issues))


def align_japanese(prepared: PreparedVocals, request: JapaneseAlignmentRequest, *, backend,
                   config: JapaneseConfig = JapaneseConfig(), cache=None, cancel=None, event_sink=None):
    request.__post_init__()
    prepared.__post_init__()
    checkpoint(cancel,event_sink,'prepare')
    texts,tokens,ranges,problems=_profile(request)
    execution={'adoption_policy':'proposed_endpoints_only; null_means_no_new_suggestion','route':'japanese','profile':'ja-hepburn/2','ctc':CTC_VERSION,'config':asdict(config),
               'pcm_id':prepared.pcm_id,'original_id':prepared.original_id,'segment_source':prepared.segment_source,
               'auto_conflict':prepared.segment_source=='manual' and prepared.analysis.is_vocal is False,
               'request_id':cache_key(asdict(request)), 'input':asdict(request), 'score_definition':'frame log probabilities + one-time anchor entry penalties + per-frame soft costs'}
    reason=None
    if prepared.segment_source=='automatic' and prepared.analysis.is_vocal is False:
        reason='no_vocal_evidence'
    elif not tokens:
        reason='no_emittable_targets'
    checkpoint(cancel,event_sink,'concat')
    concat=concatenate_cores(prepared.vocals,prepared.segments)
    execution['pieces']=[asdict(p) for p in concat.pieces]
    if concat.audio.samples.shape[1]<400:
        reason=reason or 'insufficient_audio'
    spans=None
    if reason is None:
        identity=getattr(backend,'cache_identity',None)
        identity=identity() if callable(identity) else identity
        execution['backend']=identity
        acoustic_key=cache_key({'pcm':prepared.pcm_id,'pieces':execution['pieces'],'backend':identity,'algorithm':ACOUSTIC_VERSION,
                                'resampling':'mean-stereo;ffmpeg-swr32-phase10-cutoff097-pad64-ceil-v1'}) if identity is not None else None
        lp=cache.get_array(acoustic_key) if cache is not None and acoustic_key is not None else None
        if lp is None:
            lp=backend.emissions(concat.audio,cancel=cancel,event_sink=event_sink)
            if cache is not None and acoustic_key is not None:
                cache.set_array(acoustic_key,lp)
        if lp.shape != (max(0,(concat.audio.samples.shape[1]-400)//320+1),32):
            raise ValueError('acoustic artifact frame/vocabulary shape mismatch')
        execution['acoustic_id']=acoustic_key
        from .activity import frame_activity
        rest, activity = frame_activity(prepared, concat, len(lp)) if config.rest_blank or config.rest_token else (None, None)
        priors = {'line_blank': config.line_blank, 'rest_blank': config.rest_blank, 'rest_token': config.rest_token}
        execution['soft_costs'] = {**priors, 'per_second': {'line_blank': int(config.line_blank),
            'rest_blank': 4 * int(config.rest_blank), 'rest_token': 25 * int(config.rest_token)}, 'activity': activity}
        token_lines = [line.line_id for line in request.lines for u in line.units for _ in range(*ranges[u.unit_id])]
        decode_options = {**priors, 'token_lines': token_lines, 'frame_seconds': .02, 'rest': rest}

        try:
            decode_key=cache_key({'stage':'japanese-decode-v1','acoustic':acoustic_key,
                                  'request':asdict(request),'profile':'ja-hepburn/2','ctc':CTC_VERSION, 'soft_costs': execution['soft_costs']}) if acoustic_key else None
            execution['decode_id'] = decode_key
            cached=cache.get_json(decode_key) if cache is not None and decode_key else None
            if cached is not None:
                try:
                    if cached['schema']!='ctc-path-v1':raise ValueError('schema')
                    spans=[tuple(s) for s in cached['spans']];score=cached['score']
                    if len(spans)!=len(tokens) or any(not 0<=a<b<=len(lp) for a,b in spans) or not np.isfinite(score):
                        raise ValueError('invalid cached CTC path')
                except (KeyError,TypeError,ValueError):cached=None
            if cached is None:
                spans,score=decode(lp,tokens, **decode_options, max_decode_memory_bytes=config.max_decode_memory_bytes,cancel=cancel,event_sink=event_sink)
                if cache is not None and decode_key:
                    cache.set_json(decode_key,{'schema':'ctc-path-v1','spans':spans,'score':score})
            execution['first_pass_score']=score
        except NoFeasiblePath:
            reason='no_feasible_path'
    if spans is None:
        units=tuple(JapaneseUnitResult(u.unit_id,l.line_id,u.reading,texts[u.unit_id],Timing(),Timing(),
                    issues=(JapaneseIssue(u.unit_id,reason),)) for l in request.lines for u in l.units)
    else:
        units=_units(request,texts,ranges,problems,spans,concat)
    calibration={'mode':config.hint_mode,'adopted':'first_pass','reason':'hints_off'}
    if spans is not None and config.hint_mode=='auto':
        from .calibration import calibrate
        checkpoint(cancel,event_sink,'calibration')
        anchors,calibration=calibrate(request,units,ranges,concat)
        calibration['mode']='auto'
        execution['first_pass_id']=cache_key({'spans':spans,'request':execution['request_id']})
        if anchors:
            try:
                second,score=decode(lp,tokens,anchors=anchors,**decode_options,max_decode_memory_bytes=config.max_decode_memory_bytes,
                                    cancel=cancel,event_sink=event_sink)
                units=_units(request,texts,ranges,problems,second,concat)
                execution['second_pass_id']=cache_key({'spans':second,'anchors':[asdict(a) for a in anchors]})
                execution['second_pass_score']=score
                calibration['adopted']='second_pass'
            except NoFeasiblePath:
                calibration['reason']='second_pass_infeasible'
    if spans is not None and config.completion:
        from .completion import complete_units, VERSION as COMPLETION_VERSION
        checkpoint(cancel,event_sink,'completion')
        units=complete_units(request,units,concat,prepared.original_duration)
        execution['completion']=COMPLETION_VERSION
    issues=[JapaneseIssue('request','partial_targets')] if any(v!='profile_hold' for v in problems.values()) else []
    checkpoint(cancel)
    if event_sink is not None:event_sink(ProgressEvent('japanese','completed'))
    return _result(request,units,execution,calibration,issues)
