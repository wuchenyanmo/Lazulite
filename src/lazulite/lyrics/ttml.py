"""Minimal safe TTML projection, shared by line and Japanese routes."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from lazulite.models.lyrics import LyricDocument,LyricLine
from lazulite.models.japanese import JapaneseAlignmentRequest,JapaneseLine,JapaneseUnit,StartHint


class TtmlFormatError(ValueError):
    pass


def _local_name(name):return name.rsplit('}',1)[-1]

def _attribute(element,name):
    return next((v for k,v in element.attrib.items() if _local_name(k)==name),None)


def _parse_time(value, *, field):
    v=value.strip()
    try:
        if re.fullmatch(r'\d+(?:\.\d+)?ms',v):seconds=float(v[:-2])/1000
        elif re.fullmatch(r'\d+(?:\.\d+)?s?',v):seconds=float(v.rstrip('s'))
        elif re.fullmatch(r'(?:\d+:)?\d+:[0-5]\d(?:\.\d+)?',v):
            parts=v.split(':')
            if len(parts)==3 and not 0<=int(parts[1])<60:raise ValueError
            seconds=sum(float(x)*60**i for i,x in enumerate(reversed(parts)))
        else:raise ValueError
        if not isfinite(seconds) or seconds<0:raise ValueError
        return seconds
    except ValueError as error:
        raise TtmlFormatError(f'不支持的 {field} 时间表达式：{value!r}；不支持 tick/frame/相对时间') from error


_EXCLUDED={'x-bg','background','x-translation','translation','x-transliteration','x-roman','x-romaji','transliteration'}

def _excluded(element):
    roles=(_attribute(element,'role') or '').split()
    return bool(_EXCLUDED.intersection(roles)) or _local_name(element.tag) in {'rt','rp'}


def _ruby(element):
    return _attribute(element,'ruby') or ('container' if _local_name(element.tag)=='ruby' else None)


def _visible(element):
    if _excluded(element) or _ruby(element) in {'text','textContainer','delimiter'}:return ''
    return (element.text or '')+''.join(_visible(c)+(c.tail or '') for c in element)


@dataclass(frozen=True)
class _Projection:
    line_id: str
    text: str
    units: tuple[JapaneseUnit,...]
    begin: float | None
    end: float | None
    agent: str | None


def _project_paragraph(element,line_id):
    from lazulite.alignment.japanese.kana import is_kana_text
    text='';units=[]
    def append(value,reading=None,source='unknown',language='unknown'):
        nonlocal text
        if not value:return
        start=len(text);text+=value
        if not value.strip():return
        if reading is None:
            reading=value if is_kana_text(value) else ''
            source='literal_kana' if reading else 'unknown'
            language='ja' if reading else 'unknown'
        units.append(JapaneseUnit(f'{line_id}:unit:{len(units)}',(start,len(text)),reading,language,source))
    def walk(node):
        for name in ('begin','end','dur'):
            value=_attribute(node,name)
            if value is not None:_parse_time(value,field=name)
        if _excluded(node) or _ruby(node) in {'text','textContainer','delimiter'}:return
        if _ruby(node)=='container':
            bases=[c for c in node.iter() if c is not node and (_ruby(c)=='base' or _local_name(c.tag)=='rb')]
            readings=[c for c in node.iter() if c is not node and (_ruby(c)=='text' or _local_name(c.tag)=='rt')]
            if len(bases)==1 and len(readings)==1:
                base=_visible(bases[0]);reading=''.join(readings[0].itertext())
                # Ruby reading is explicitly associated; Latin cannot masquerade as kana.
                language='ja' if is_kana_text(reading) else 'latin' if re.fullmatch(r"[A-Za-z' ]+",reading) else 'unknown'
                append(node.text or '')
                for child in node:
                    if child is bases[0]:append(base,reading if language!='unknown' else '', 'imported_ruby',language)
                    elif child is not readings[0]:append(_visible(child))
                    append(child.tail or '')
            else:
                append(_visible(node),'','ambiguous_ruby','unknown')
            return
        append(node.text or '')
        for child in node:
            walk(child)
            append(child.tail or '')
    walk(element)
    begin=_attribute(element,'begin');end=_attribute(element,'end')
    a=_parse_time(begin,field=f'{line_id} begin') if begin is not None else None
    b=_parse_time(end,field=f'{line_id} end') if end is not None else None
    if a is not None and b is not None and a>b:raise TtmlFormatError('line begin exceeds end')
    return _Projection(line_id,text,tuple(units),a,b,_attribute(element,'agent'))


def _read(path):
    try:
        data=Path(path).read_bytes();root=ET.fromstring(data)
    except (OSError,ET.ParseError) as error:raise TtmlFormatError(f'无法读取 TTML：{path}') from error
    snapshot=hashlib.sha256(data).hexdigest()[:16]
    rows=[];ids=set()
    def visit(node):
        if _excluded(node):return
        tag=_local_name(node.tag)
        for name in ('begin','end','dur'):
            value=_attribute(node,name)
            if value is not None:_parse_time(value,field=name)
        if _attribute(node,'timeContainer') not in {None,'par'}:
            raise TtmlFormatError('不支持相对时间容器')
        if tag in {'tt','body','div'} and any(_attribute(node,k) is not None for k in ('begin','end','dur')):
            raise TtmlFormatError('不支持带时间的父容器')
        if tag=='p':
            line_id=_attribute(node,'key') or _attribute(node,'id') or f'snapshot-{snapshot}-line-{len(rows)}'
            if line_id in ids:raise TtmlFormatError(f'TTML 行 ID 重复：{line_id}')
            ids.add(line_id)
            row=_project_paragraph(node,line_id)
            if row.text.strip():rows.append(row)
        else:
            for child in node:visit(child)
    visit(root)
    if not rows:raise TtmlFormatError('TTML 中没有可对齐的正文行')
    return rows,_attribute(root,'lang')


def read_ttml_for_alignment(path, *, language=None):
    rows,lang=_read(path)
    return LyricDocument(tuple(LyricLine(r.line_id,r.text,r.begin,r.end,r.agent) for r in rows),language=language or lang)


def read_ttml_for_japanese(path):
    rows,_=_read(path)
    return JapaneseAlignmentRequest(tuple(JapaneseLine(r.line_id,r.text,r.units,
        StartHint(r.begin,'imported') if r.begin is not None else None) for r in rows))
