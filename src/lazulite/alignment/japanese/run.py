"""Independent Japanese stages, retaining every requested object ID."""
from dataclasses import asdict, dataclass
from lazulite.audio.concat import concatenate_cores
from lazulite.events import checkpoint, ProgressEvent
from lazulite.models.audio import PreparedVocals
from lazulite.models.japanese import (
    Endpoint, Timing, JapaneseAlignmentRequest, JapaneseAlignmentResult, JapaneseUnitResult, JapaneseIssue,
)
from .stages import AcousticResult, DecodeResult
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


def _profile(request, vocabulary, blank, special_tokens=(), encoder=None):
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
            encoded = encoder(text, vocabulary) if encoder is not None and text else ([vocabulary[c] for c in text] if text and all(c in vocabulary and c not in special_tokens and c not in '| ' and vocabulary[c] != blank for c in text) else None)
            if encoded:
                if any(type(t) is not int or t == blank or t not in vocabulary.values() for t in encoded):
                    raise ValueError('encoder produced invalid target IDs')
                tokens.extend(encoded)
            else:
                problems[u.unit_id] = 'profile_hold' if u.reading_language=='ja' and is_hold(u.reading) else 'unresolved_reading'
            ranges[u.unit_id] = (begin,len(tokens))
    return texts,tokens,ranges,problems


def _intervals(request, spans, ranges, frames):
    raw = {}
    sources = {}
    for line in request.lines:
        rows = []
        for u in line.units:
            a,b = ranges[u.unit_id]
            rows.append([int(round(frames.time(spans[a][0])*1000)),int(round(frames.time(spans[b-1][1])*1000))] if b>a else None)
        i=0
        while i<len(rows):
            j=i+1
            while j<len(rows) and is_hold(line.units[j].reading) and line.units[j].reading_language=='ja' and rows[j] is None:
                j+=1
            if j>i+1 and rows[i] is not None and not is_hold(line.units[i].reading):
                a,b=rows[i]
                nxt=next((r[0] for r in rows[j:] if r is not None),None)
                end=b+(min(800,max(0,nxt-b)) if nxt is not None else 300)
                end=max(end,a+frames.seconds*1000*(j-i))
                if nxt is not None and nxt>=a+frames.seconds*1000*(j-i):
                    end=min(end,nxt)
                step=(end-a)/(j-i)
                for k in range(i,j):
                    rows[k]=[int(round(a+step*(k-i))),int(round(a+step*(k-i+1))) if k<j-1 else int(end)]
                    sources[line.units[k].unit_id]='profile_hold'
            i=max(j,i+1)
        for u,row in zip(line.units,rows):
            raw[u.unit_id] = None if row is None else (row[0]/1000,row[1]/1000)
    return raw,sources


def _units(request,texts,ranges,problems,spans,concat,frames):
    intervals,sources=_intervals(request,spans,ranges,frames)
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
                if b>concat.duration:
                    issue.append(JapaneseIssue(u.unit_id,'profile_outside_audio',{'end':b}))
                    b=concat.duration
                if b>a:
                    fragments=concat.fragments(a,b)
                    source=sources.get(u.unit_id,'acoustic_path')
                    timing=Timing(Endpoint(fragments[0].start,source,(u.unit_id,)), Endpoint(fragments[-1].end,source,(u.unit_id,)))
            cross=len(fragments)>1
            if cross:
                issue.append(JapaneseIssue(u.unit_id,'crosses_removed_gap'))
            a,b=ranges[u.unit_id]
            chars=tuple({'character':texts[u.unit_id][k-a] if len(texts[u.unit_id]) == b-a else None, 'frames':s,
                         'concat_start':frames.time(s[0]),'concat_end':frames.time(s[1])} for k,s in enumerate(spans[a:b],a))
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



def compute_acoustics(prepared: PreparedVocals, *, backend, include_activity=False, cancel=None, event_sink=None):
    prepared.__post_init__()
    checkpoint(cancel,event_sink,'concat')
    segments=() if prepared.segment_source=='automatic' and prepared.analysis.is_vocal is False else prepared.segments
    concat=concatenate_cores(prepared.vocals,segments,sample_rate=backend.frames.sample_rate)
    lp=backend.emissions(concat.audio,cancel=cancel,event_sink=event_sink)
    if lp.shape != (backend.frames.count(concat.audio.samples.shape[1]),backend.vocabulary_size):
        raise ValueError('acoustic output shape does not match backend frame contract')
    rest,activity=None,None
    if include_activity:
        from .activity import frame_activity
        rest,activity=frame_activity(prepared,concat.mapping,len(lp),backend.frames)
    checkpoint(cancel)
    return AcousticResult(lp,dict(backend.vocab),backend.blank,backend.frames,concat.mapping,
        prepared.original_duration,rest,{'backend':backend.description,'activity':activity,
        'original_id':prepared.original_id,'segment_source':prepared.segment_source,
        'no_vocal':prepared.segment_source=='automatic' and prepared.analysis.is_vocal is False,
        'auto_conflict':prepared.segment_source=='manual' and prepared.analysis.is_vocal is False},
        tuple(backend.special_tokens))


def decode_japanese(acoustic: AcousticResult, request: JapaneseAlignmentRequest, *,
                    config=JapaneseConfig(), prepared=None, encoder=None, cancel=None, event_sink=None):
    request.__post_init__()
    acoustic.__post_init__()
    checkpoint(cancel,event_sink,'decode')
    texts,tokens,ranges,problems=_profile(request,acoustic.vocabulary,acoustic.blank_id,acoustic.special_tokens,encoder)
    concat,frames,lp=acoustic.mapping,acoustic.frames,acoustic.log_probabilities
    execution={'adoption_policy':'proposed_endpoints_only; null_means_no_new_suggestion','route':'japanese',
        'profile':'ja-hepburn/2' if encoder is None else 'custom-encoder','ctc':CTC_VERSION,'config':asdict(config),
        'input':asdict(request),'pieces':[asdict(p) for p in concat.pieces],**acoustic.description,
        'frames':asdict(frames),'score_definition':'frame log probabilities + one-time anchor entry penalties + per-frame soft costs'}
    reason='no_vocal_evidence' if acoustic.description.get('no_vocal') else 'no_emittable_targets' if not tokens else 'insufficient_audio' if not len(lp) else None
    spans=first=None
    calibration={'mode':config.hint_mode,'adopted':'first_pass','reason':'hints_off'}
    rest=acoustic.rest
    if config.rest_blank or config.rest_token:
        if rest is None:
            if prepared is None: raise ValueError('rest costs require frame activity or complete vocals')
            prepared.__post_init__()
            expected=concatenate_cores(prepared.vocals,prepared.segments,sample_rate=frames.sample_rate).mapping
            if expected!=concat or prepared.original_duration!=acoustic.original_duration:
                raise ValueError('activity vocals do not match acoustic coordinates')
            from .activity import frame_activity
            rest,activity=frame_activity(prepared,concat,len(lp),frames)
            execution['activity']=activity
    priors={k:getattr(config,k) for k in ('line_blank','rest_blank','rest_token')}
    execution['soft_costs']={**priors,'per_second':{'line_blank':int(config.line_blank),
        'rest_blank':4*int(config.rest_blank),'rest_token':25*int(config.rest_token)},
        'activity':execution.get('activity')}
    options={**priors,'blank':acoustic.blank_id,
        'token_lines':[l.line_id for l in request.lines for u in l.units for _ in range(*ranges[u.unit_id])],
        'frame_seconds':frames.seconds,'rest':rest,'max_decode_memory_bytes':config.max_decode_memory_bytes,
        'cancel':cancel,'event_sink':event_sink}
    if reason is None:
        try:
            spans,score=decode(lp,tokens,**options)
            first=tuple(spans)
            execution['first_pass_score']=score
        except NoFeasiblePath:
            reason='no_feasible_path'
    if spans is not None and config.hint_mode=='auto':
        from .calibration import calibrate
        checkpoint(cancel,event_sink,'calibration')
        units=_units(request,texts,ranges,problems,spans,concat,frames)
        anchors,calibration=calibrate(request,units,ranges,concat,frames)
        calibration['mode']='auto'
        if anchors:
            try:
                spans,score=decode(lp,tokens,anchors=anchors,**options)
                execution['second_pass_score']=score
                calibration['adopted']='second_pass'
            except NoFeasiblePath:
                calibration['reason']='second_pass_infeasible'
    checkpoint(cancel)
    return DecodeResult(request,tuple(tokens),dict(acoustic.vocabulary),acoustic.blank_id,acoustic.special_tokens,lp.shape[1],ranges,texts,problems,first,tuple(spans) if spans is not None else None,
        len(lp),frames,concat,acoustic.original_duration,execution,calibration,reason)


def finalize_japanese(decoded: DecodeResult, *, completion=True, cancel=None,event_sink=None):
    decoded.__post_init__()
    checkpoint(cancel,event_sink,'completion')
    request=decoded.request
    execution={**decoded.execution,'config':{**decoded.execution['config'],'completion':completion}}
    if decoded.spans is None:
        units=tuple(JapaneseUnitResult(u.unit_id,l.line_id,u.reading,decoded.texts[u.unit_id],Timing(),Timing(),
            issues=(JapaneseIssue(u.unit_id,decoded.reason),)) for l in request.lines for u in l.units)
    else:
        units=_units(request,decoded.texts,decoded.ranges,decoded.problems,decoded.spans,decoded.mapping,decoded.frames)
        if completion:
            from .completion import complete_units, VERSION as COMPLETION_VERSION
            units=complete_units(request,units,decoded.mapping,decoded.original_duration)
            execution['completion']=COMPLETION_VERSION
    issues=[JapaneseIssue('request','partial_targets')] if any(v!='profile_hold' for v in decoded.problems.values()) else []
    checkpoint(cancel)
    if event_sink is not None:event_sink(ProgressEvent('japanese','completed'))
    return _result(request,units,execution,dict(decoded.calibration),issues)


def align_japanese(prepared: PreparedVocals | None, request: JapaneseAlignmentRequest, *, backend=None,
                   acoustic: AcousticResult | None=None, decoded: DecodeResult | None=None,
                   config=JapaneseConfig(), encoder=None, cancel=None,event_sink=None):
    if decoded is not None:
        if acoustic is not None or prepared is not None or backend is not None or encoder is not None:
            raise ValueError('decoded result is a complete input; conflicting earlier-stage sources')
        if request != decoded.request: raise ValueError('decoded targets differ from request')
        expected={k:v for k,v in asdict(config).items() if k!='completion'}
        actual={k:v for k,v in decoded.execution['config'].items() if k!='completion'}
        if expected!=actual: raise ValueError('decode configuration differs; decode again explicitly')
    else:
        if acoustic is None:
            if prepared is None or backend is None: raise ValueError('complete vocals and acoustic backend required')
            acoustic=compute_acoustics(prepared,backend=backend,include_activity=config.rest_blank or config.rest_token,
                                       cancel=cancel,event_sink=event_sink)
        elif backend is not None:
            raise ValueError('acoustic result conflicts with acoustic backend')
        decoded=decode_japanese(acoustic,request,config=config,prepared=prepared,encoder=encoder,
                                cancel=cancel,event_sink=event_sink)
    return finalize_japanese(decoded,completion=config.completion,cancel=cancel,event_sink=event_sink)
