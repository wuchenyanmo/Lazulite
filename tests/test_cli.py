import json
import pytest
from lazulite.cli import build_parser,main


def test_direct_vocals_and_optional_stage_models():
    p=build_parser()
    args=p.parse_args(['align','lyrics.ttml','--vocals','vocals.wav','--backend','whisper','--model-path','model','--output','out.json'])
    assert str(args.vocals)=='vocals.wav' and args.audio is None
    args=p.parse_args(['align-japanese','--acoustic-input','a.npz','--targets','t.json','--output','out.json'])
    assert args.model_path is None
    args=p.parse_args(['forced-align','song.wav','text.txt','--backend','qwen','--model-path','model','--output','out.json'])
    assert args.line_id=='line-0'


@pytest.mark.parametrize('option',['--cache','--revision','--prepared','--separator-revision'])
def test_removed_options_are_not_accepted(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(['align','lyrics.ttml','--vocals','v.wav','--output','o.json',option,'old'])


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
