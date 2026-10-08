"""命令行解析与错误边界不需要加载模型。"""

from __future__ import annotations

from lazulite.cli import build_parser, main


def test_cli_exposes_the_documented_commands_without_importing_models() -> None:
    parser = build_parser()
    lyrics = parser.parse_args(["lyrics", "search", "song.m4a", "--output", "candidates.json"])
    forced = parser.parse_args([
        "forced-align", "clip.wav", "confirmed.txt", "--backend", "qwen",
        "--model-path", "model", "--revision", "abc123", "--output", "fine.json",
    ])

    assert lyrics.command == "lyrics"
    assert lyrics.lyrics_command == "search"
    assert forced.command == "forced-align"
    assert forced.revision == "abc123"
    assert not hasattr(forced, "full_distributions")


def test_cli_returns_a_concise_nonzero_error_for_missing_input(tmp_path) -> None:
    output = tmp_path / "result.json"

    assert main(["lyrics", "search", "missing.m4a", "--output", str(output)]) == 2
    assert not output.exists()


def test_prepared_inputs_are_explicit_and_invalid_request_precedes_model(tmp_path):
    import json
    import pytest
    parser=build_parser()
    args=parser.parse_args(['align','lyrics.ttml','--prepared','manifest.json','--backend','whisper','--model-path','absent','--revision','r','--output','out.json'])
    assert args.audio is None and str(args.lyrics)=='lyrics.ttml'
    with pytest.raises(SystemExit):
        parser.parse_args(['align-japanese','--audio','song.wav','--prepared','manifest.json','--targets','units.json','--model-path','absent','--output','out.json'])
    target=tmp_path/'invalid.json';target.write_text(json.dumps({'schema':'wrong'}))
    assert main(['align-japanese','--prepared','absent','--targets',str(target),'--model-path','absent','--output',str(tmp_path/'out.json')])==2
    assert not (tmp_path/'out.json').exists()
