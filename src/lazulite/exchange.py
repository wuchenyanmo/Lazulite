"""Explicit caller-selected files. No discovery, keys, manifests or automatic reuse."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .audio.concat import ConcatMap, Piece
from .alignment.japanese.stages import AcousticResult, DecodeResult, FrameMap
from .models.japanese import JapaneseAlignmentRequest


def _mapping(value):
    return ConcatMap(value['sample_rate'],value['sample_count'],tuple(Piece(**p) for p in value['pieces']))


def write_acoustic(path, result):
    result.__post_init__()
    metadata={k:getattr(result,k) for k in ('vocabulary','blank_id','original_duration','description','special_tokens')}
    metadata.update(schema='lazulite-acoustic-v1',frames=asdict(result.frames),mapping=asdict(result.mapping),
                    rest_present=result.rest is not None,numerical_definition='normalized natural log probabilities')
    with Path(path).open('wb') as stream:
        np.savez(stream,metadata=np.asarray(json.dumps(metadata,allow_nan=False)),
            log_probabilities=result.log_probabilities,rest=result.rest if result.rest is not None else np.empty(0))


def read_acoustic(path):
    with np.load(path,allow_pickle=False) as data:
        value=json.loads(str(data['metadata']))
        if value['schema']!='lazulite-acoustic-v1' or value['numerical_definition']!='normalized natural log probabilities':
            raise ValueError('unsupported acoustic format')
        return AcousticResult(data['log_probabilities'],value['vocabulary'],value['blank_id'],FrameMap(**value['frames']),
            _mapping(value['mapping']),value['original_duration'],data['rest'] if value['rest_present'] else None,
            value['description'],tuple(value['special_tokens']))


def write_decode(path, result):
    result.__post_init__()
    Path(path).write_text(json.dumps({'schema':'lazulite-decode-v1',**asdict(result)},ensure_ascii=False,allow_nan=False),encoding='utf-8')


def read_decode(path):
    v=json.loads(Path(path).read_text('utf-8'))
    if v.pop('schema')!='lazulite-decode-v1': raise ValueError('unsupported decode format')
    if v['execution']['input'] != v['request']:
        raise ValueError('decoded execution and target request differ')
    v['request']=JapaneseAlignmentRequest.from_dict({'schema':'lazulite-japanese-request-v1',**v['request']})
    v['execution']['input']=asdict(v['request'])
    v['frames']=FrameMap(**v['frames']);v['mapping']=_mapping(v['mapping'])
    v['special_tokens']=tuple(v['special_tokens'])
    v['tokens']=tuple(v['tokens']);v['ranges']={k:tuple(x) for k,x in v['ranges'].items()}
    for key in ('first_pass','spans'):
        if v[key] is not None: v[key]=tuple(tuple(x) for x in v[key])
    return DecodeResult(**v)


def write_transcriptions(path, stages):
    Path(path).write_text(json.dumps({'schema':'lazulite-transcriptions-v1','stages':{k:asdict(v) for k,v in stages.items()}},
                                   ensure_ascii=False,allow_nan=False),encoding='utf-8')


def read_transcriptions(path):
    from .transcription.run import TranscriptionStage, TranscriptionConfig
    from .transcription.base import TranscriberCapabilities,TranscriptionRequest
    from .transcription.evidence import EvidenceConfig
    from .models.audio import VocalSegment
    from .models.transcription import BackendInfo,TranscriptionResult,transcription_segment_from_dict
    value=json.loads(Path(path).read_text('utf-8'))
    if value['schema']!='lazulite-transcriptions-v1': raise ValueError('unsupported transcription format')
    stages={}
    for k,v in value['stages'].items():
        config={**v['config'],'evidence':EvidenceConfig(**v['config']['evidence'])}
        stages[k]=TranscriptionStage(tuple(VocalSegment(**s) for s in v['segments']),TranscriptionRequest(**v['request']),
            TranscriptionConfig(**config),TranscriberCapabilities(**v['capabilities']),
            TranscriptionResult(BackendInfo(**v['result']['backend']),tuple(transcription_segment_from_dict(s) for s in v['result']['segments'])))
    return stages
