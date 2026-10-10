"""Bounded proposals derived from local retained vocal pieces; raw path is immutable."""
from dataclasses import replace
from lazulite.models.japanese import Endpoint,Timing,JapaneseIssue
from .kana import split_morae,is_kana_text

VERSION='local-piece-completion-v1'


def complete_units(request, units, concat, duration):
    result=list(units)
    index={u.unit_id:i for i,u in enumerate(result)}
    pieces=[(p.source_start/concat.sample_rate,min(duration,p.source_end/concat.sample_rate)) for p in concat.pieces]
    def set_timing(i,start,end,source,basis):
        if start is not None and end is not None and not start<end:
            return
        result[i]=replace(result[i],proposed_timing=Timing(
            Endpoint(start,source,basis) if start is not None else None,
            Endpoint(end,source,basis) if end is not None else None),
            issues=result[i].issues+(JapaneseIssue(result[i].unit_id,source),))
    for line in request.lines:
        indices=[index[u.unit_id] for u in line.units]
        # Cross-gap fragments: constrain by already continuous adjacent evidence.
        for k,i in enumerate(indices):
            unit=result[i]
            if not unit.crosses_removed_gap:
                continue
            prev=result[indices[k-1]].proposed_timing.end if k else None
            nxt=result[indices[k+1]].proposed_timing.start if k+1<len(indices) else None
            candidates=[]
            for f in unit.fragments:
                a=max(f.start,prev.time if prev else 0)
                b=min(f.end,nxt.time if nxt else duration)
                if a<b:
                    candidates.append((a,b))
            if len(candidates)==1:
                set_timing(i,*candidates[0],'fragment_selected',tuple(x.unit_id for x in result[max(0,i-1):i+2]))
        # Missing runs may only use a positive interval within one retained piece.
        k=0
        while k<len(indices):
            i=indices[k];t=result[i].proposed_timing
            if t.start is not None or t.end is not None or result[i].crosses_removed_gap:
                k+=1;continue
            end=k+1
            while end<len(indices):
                u=result[indices[end]]
                if u.proposed_timing.start is not None or u.proposed_timing.end is not None or u.crosses_removed_gap:
                    break
                end+=1
            left=result[indices[k-1]].proposed_timing.end if k else None
            right=result[indices[end]].proposed_timing.start if end<len(indices) else None
            if left is not None and right is not None and left.time<right.time and any(a<=left.time<right.time<=b for a,b in pieces):
                weights=[len(split_morae(u.reading)) if u.reading_language=='ja' and is_kana_text(u.reading) else 1 for u in line.units[k:end]]
                total=sum(weights);offset=0
                basis=(line.units[k-1].unit_id,line.units[end].unit_id)
                for j,w in enumerate(weights,k):
                    a=left.time+(right.time-left.time)*offset/total
                    offset+=w
                    b=left.time+(right.time-left.time)*offset/total
                    set_timing(indices[j],a,b,'interpolated',basis)
            k=end
        # One known endpoint identifies a local piece; never extend to song bounds.
        for k,i in enumerate(indices):
            u=result[i];t=u.proposed_timing
            if (t.start is None)==(t.end is None) or u.crosses_removed_gap:
                continue
            if t.start is not None:
                candidates=[(a,b) for a,b in pieces if a<=t.start.time<b]
                if len(candidates)==1:
                    a,b=candidates[0]
                    nxt=result[indices[k+1]].proposed_timing.start if k+1<len(indices) else None
                    value=min(b,nxt.time) if nxt is not None and a<nxt.time<=b else b
                    if value>t.start.time:
                        result[i]=replace(u,proposed_timing=Timing(t.start,Endpoint(value,'adjacent' if nxt and value==nxt.time else 'acoustic_piece',(line.units[k+1].unit_id,) if nxt else (u.unit_id,))))
            else:
                candidates=[(a,b) for a,b in pieces if a<t.end.time<=b]
                if len(candidates)==1:
                    a,b=candidates[0]
                    prev=result[indices[k-1]].proposed_timing.end if k else None
                    value=max(a,prev.time) if prev is not None and a<=prev.time<b else a
                    if value<t.end.time:
                        result[i]=replace(u,proposed_timing=Timing(Endpoint(value,'adjacent' if prev and value==prev.time else 'acoustic_piece',(line.units[k-1].unit_id,) if prev else (u.unit_id,)),t.end))
    return tuple(result)
