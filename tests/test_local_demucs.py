import pytest
from lazulite.audio.separation import DemucsConfig, DemucsSeparator


def test_local_repo_missing_files_never_uses_remote(tmp_path, monkeypatch):
    pytest.importorskip('demucs')
    import demucs.pretrained
    calls = []
    monkeypatch.setattr(demucs.pretrained, 'get_model', lambda *a, **k: calls.append((a, k)))
    separator = DemucsSeparator(DemucsConfig(device='cpu', model_repo=str(tmp_path)))
    with pytest.raises(FileNotFoundError, match='模型缺失'): separator._load()
    assert calls == []


def test_local_repo_does_not_change_quality_parameters():
    from dataclasses import asdict
    plain = asdict(DemucsConfig(device='cpu'))
    local = asdict(DemucsConfig(device='cpu', model_repo='/relocated/models'))
    plain.pop('model_repo'); local.pop('model_repo')
    assert plain == local
