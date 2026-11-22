import json
import pytest
from lazulite.cli import build_parser,main


def test_direct_vocals_and_optional_stage_models():
    p=build_parser()
    args=p.parse_args(['align','lyrics.ttml','--vocals','vocals.wav','--backend','whisper','--model-path','model','--output','out.json'])
    assert str(args.vocals)=='vocals.wav' and args.audio is None
    args=p.parse_args(['align-japanese','--acoustic-input','a.npz','--targets','t.json','--output','out.json'])
    assert args.model_path is None
    args=p.parse_args(['align-japanese','--audio','song.wav','--ttml','lyrics.ttml','--model-path','model','--separator','bs-karaoke','--bs-karaoke-repo','repo','--no-latin-normalize','--strip-tone-digits','--output','out.json'])
    assert args.separator=='bs-karaoke' and args.vocal_model_repo=='repo'
    assert args.latin_normalize is False and args.strip_tone_digits is True


def test_align_japanese_cli_wires_open_backend_separator_and_input_flags(monkeypatch, tmp_path):
    import lazulite.audio.bs_karaoke as bs_karaoke
    import lazulite.audio.io as audio_io
    import lazulite.audio.preparation as preparation
    import lazulite.audio.separation as separation
    import lazulite.alignment.japanese.backend as ja_backend
    import lazulite.alignment.japanese.run as ja_run
    from lazulite.models.japanese import JapaneseAlignmentResult

    target=tmp_path/'targets.json'
    target.write_text(json.dumps({
        'schema':'lazulite-japanese-request-v1',
        'lines':[{'line_id':'l','text':'あ','units':[{'unit_id':'u','source_range':[0,1],'reading':'あ','reading_language':'ja','reading_source':'user'}]}],
    }))
    recorded={}
    class _Sep:
        def __init__(self,config): recorded['sep_config']=config
        def close(self): recorded['sep_closed']=True
    class _Backend:
        def close(self): recorded['backend_closed']=True
    monkeypatch.setattr(audio_io,'decode_audio',lambda *a,**k:'raw-audio')
    monkeypatch.setattr(preparation,'prepare_vocals',lambda audio,*,separator: recorded.update(prepared_with=separator) or 'prepared')
    monkeypatch.setattr(bs_karaoke,'BSKaraokeSeparator',_Sep)
    monkeypatch.setattr(separation,'DemucsSeparator',_Sep)
    monkeypatch.setattr(ja_backend,'open_backend',lambda path,device='cpu': recorded.update(backend_args=(path,device)) or _Backend())
    monkeypatch.setattr(ja_run,'compute_acoustics',lambda prepared,*,backend: 'acoustic')
    def _decode(acoustic,request,*,config):
        recorded['decode_config']=config
        return 'decoded'
    monkeypatch.setattr(ja_run,'decode_japanese',_decode)
    monkeypatch.setattr(ja_run,'finalize_japanese',lambda decoded,*,completion=True: JapaneseAlignmentResult(() ,'complete',(),{}, {},{}))
    out=tmp_path/'out.json'
    assert main(['align-japanese','--audio',str(tmp_path/'a.wav'),'--targets',str(target),'--model-path',str(tmp_path/'m'),'--device','cuda','--separator','bs-karaoke','--vocal-model-repo','bs-repo','--no-latin-normalize','--strip-tone-digits','--output',str(out)])==0
    assert recorded['sep_config']==bs_karaoke.BSKaraokeConfig('bs-repo',device='cuda')
    assert recorded['backend_args']==(tmp_path/'m','cuda')
    assert recorded['decode_config'].latin_normalize is False and recorded['decode_config'].strip_tone_digits is True
    assert recorded['sep_closed'] and recorded['backend_closed']
    assert main(['align-japanese','--audio',str(tmp_path/'a.wav'),'--targets',str(target),'--model-path',str(tmp_path/'m'),'--separator','bs-karaoke','--output',str(tmp_path/'err.json')])==2
    assert main(['align-japanese','--vocals',str(tmp_path/'v.wav'),'--targets',str(target),'--model-path',str(tmp_path/'m'),'--separator','bs-karaoke','--vocal-model-repo','r','--output',str(tmp_path/'err2.json')])==2


def test_cli_rejects_conflicts_and_bad_targets_without_loading_models(tmp_path):
    with pytest.raises(SystemExit): build_parser().parse_args(['prepare-vocals','v.wav'])
    with pytest.raises(SystemExit):
        build_parser().parse_args(['align-japanese','--audio','a.wav','--vocals','v.wav','--targets','t.json','--output','o.json'])
    target=tmp_path/'invalid.json';target.write_text(json.dumps({'schema':'wrong'}))
    assert main(['align-japanese','--vocals','absent','--targets',str(target),'--model-path','absent','--output',str(tmp_path/'o.json')])==2
    assert not (tmp_path/'o.json').exists()


def test_explicit_acoustic_and_decode_cli_need_no_model(tmp_path):
    from test_stages import inputs
    from fixtures.acoustic import Backend
    from lazulite.alignment.japanese.run import compute_acoustics
    from lazulite.exchange import write_acoustic
    from dataclasses import asdict
    prepared,request=inputs()
    a=tmp_path/'a.npz';t=tmp_path/'t.json';d=tmp_path/'d.json';o=tmp_path/'o.json';o2=tmp_path/'o2.json'
    write_acoustic(a,compute_acoustics(prepared,backend=Backend()))
    t.write_text(json.dumps({'schema':'lazulite-japanese-request-v1',**asdict(request)}))
    assert main(['align-japanese','--acoustic-input',str(a),'--targets',str(t),'--decode-output',str(d),'--output',str(o)])==0
    assert main(['align-japanese','--decode-input',str(d),'--targets',str(t),'--output',str(o2)])==0
    assert json.loads(o.read_text())==json.loads(o2.read_text())
    assert main(['align-japanese','--acoustic-input',str(a),'--targets',str(t),'--output',str(a)])==2
