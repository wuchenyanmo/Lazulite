"""Derive missing ends from ordered lexical starts without changing raw metadata."""
from dataclasses import replace
from lazulite.lyrics.normalize import normalize_lyric_text


def organize_timing(transcription, analysis=None):
    parents={} if analysis is None else {s.id:s for s in analysis.segments}
    def window(s):return s.request_window or parents.get(s.source_segment_id)
    indexed=list(enumerate(transcription.segments))
    indexed.sort(key=lambda pair: (window(pair[1]).start if window(pair[1]) else float('inf'),
                                  pair[1].request_window.sequence if pair[1].request_window else pair[0]))
    ordered=[s for _,s in indexed]
    duplicate=set()
    for i,current in enumerate(ordered):
        cw=window(current)
        if cw is None:continue
        for j in range(i-1,-1,-1):
            previous=ordered[j];pw=window(previous)
            if pw is None or pw.end<=cw.start:continue
            cs=next((u.start for u in current.timed_text if u.start is not None),None)
            ps=next((u.start for u in previous.timed_text if u.start is not None),None)
            # Same text alone never removes a repeated chorus at another time.
            if cs is not None and ps is not None and abs(cs-ps)<=.25 and max(pw.start,cw.start)<=cs<min(pw.end,cw.end) and normalize_lyric_text(current.text)==normalize_lyric_text(previous.text):
                duplicate.add(i);break
    result=[]
    for i,segment in enumerate(ordered):
        w=window(segment)
        if i in duplicate:
            result.append(replace(segment,issues=tuple(dict.fromkeys(segment.issues+('duplicate_padding',)))))
            continue
        units=list(segment.timed_text)
        for k,u in enumerate(units):
            if u.start is None:continue
            end=u.end;source=u.end_source
            if end is None or end<=u.start:
                successor=next((v.start for v in units[k+1:] if v.start is not None and v.start>u.start),None)
                if successor is None:
                    for j in range(i+1,len(ordered)):
                        if j in duplicate:continue
                        following=ordered[j];fw=window(following)
                        if fw is None:break
                        candidate=next((v.start for v in following.timed_text if v.start is not None),None)
                        if candidate is not None and candidate>u.start:
                            # Overlapping windows with conflicting lexical evidence are ambiguous.
                            if w is None or fw.start>=w.end:
                                successor=candidate
                            break
                if successor is not None:
                    end,source=successor,'successor_start'
                elif w is not None and w.core_end is not None and w.core_start<=u.start<w.core_end:
                    end,source=w.core_end,'acoustic_core'
            if end is not None and source in {'successor_start','acoustic_core'} and w is not None:
                if w.core_start is not None and w.core_start<=u.start<w.core_end:
                    if end>w.core_end:end,source=w.core_end,'acoustic_core'
                elif end>w.end:end,source=w.end,'request_boundary'
            if end is not None and end<=u.start:end,source=None,None
            units[k]=replace(u,end=end,start_source=u.start_source or 'model_timestamp',end_source=source)
        result.append(replace(segment,timed_text=tuple(units)))
    return replace(transcription,segments=tuple(result))
