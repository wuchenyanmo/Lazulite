"""Original-axis hint voting; adapted from MiliKara fixed fit_offset (MIT).

Our additional coverage gate uses quarters of the complete request, both before
and after removing hints in deleted gaps. No line windows or output delta shift.
"""
import statistics
from .ctc import SoftAnchor

VERSION = 'original-hints-full-request-coverage-v1'


def _covers(indices, total):
    quarter=max(1,total//4)
    return bool(indices) and min(indices)<quarter and max(indices)>=total-quarter


def calibrate(request, units, ranges, concat):
    by_id={u.unit_id:u for u in units}
    votes=[]
    excluded=[]
    for index,line in enumerate(request.lines):
        hint=line.start_hint
        if hint is None or not line.units:
            continue
        first=by_id[line.units[0].unit_id]
        start,end=first.raw_timing.start,first.raw_timing.end
        reason=None
        if hint.source=='automatic' and hint.route=='japanese':
            reason='same_route_automatic'
        elif start is None or end is None or start.source!='acoustic_path' or end.source!='acoustic_path' or start.time>=end.time or first.crosses_removed_gap:
            reason='ineligible_first_unit'
        if reason:
            excluded.append({'line_id':line.line_id,'reason':reason})
        else:
            h=int(round(hint.time*1000))
            votes.append((index,line.line_id,h,int(round(start.time*1000))-h,line.units[0].unit_id,hint.source))
    report={'version':VERSION,'votes':len(votes),'excluded':excluded,'adopted':'first_pass'}
    if not votes:
        return (),report|{'reason':'no_eligible_hints'}
    diffs=[v[3] for v in votes]
    def density(d):
        near=[x for x in diffs if abs(x-d)<=300]
        return len(near),-statistics.pstdev(near)
    center=max(diffs,key=density)
    delta=int(round(statistics.median(d for d in diffs if abs(d-center)<=300)))
    support=[v for v in votes if abs(v[3]-delta)<=300]
    ts=[v[2] for v in support];ds=[v[3] for v in support]
    mt,md=statistics.fmean(ts),statistics.fmean(ds)
    var=sum((t-mt)**2 for t in ts)
    drift=int(round(sum((t-mt)*(d-md) for t,d in zip(ts,ds))/var*(max(ts)-min(ts)))) if var>0 else 0
    report|={'delta_seconds':delta/1000,'support':len(support),'support_ratio':len(support)/len(votes),
             'drift_seconds':drift/1000,'support_line_ids':[v[1] for v in support]}
    reason=None
    if len(votes)<4:reason='insufficient_hints'
    elif len(support)/len(votes)<.6:reason='insufficient_support'
    elif not _covers([v[0] for v in support],len(request.lines)):reason='local_support'
    elif abs(drift)>400:reason='drift'
    if reason:return (),report|{'reason':reason}
    anchors=[];kept=[]
    for v in support:
        original=(v[2]+delta)*16
        mapped=concat.to_concat(original,endpoint='start')
        if mapped is None:
            excluded.append({'line_id':v[1],'reason':'hint_in_deleted_gap'})
            continue
        anchors.append(SoftAnchor(ranges[v[4]][0],mapped/320))
        kept.append(v[0])
    report['mapped_hints']=len(anchors)
    if len(anchors)<4 or not _covers(kept,len(request.lines)):
        return (),report|{'reason':'insufficient_mapped_coverage'}
    return tuple(anchors),report|{'reason':'accepted'}
