import pytest
from lazulite.audio.separation import DemucsConfig, DemucsSeparator


def test_local_repo_missing_model_never_uses_remote(tmp_path, monkeypatch):
    pytest.importorskip('demucs')
    from demucs.repo import ModelLoadingError, RemoteRepo

    def unexpected_remote(*args, **kwargs):
        pytest.fail('local model lookup must not use the remote repository')

    monkeypatch.setattr(RemoteRepo, 'get_model', unexpected_remote)
    separator = DemucsSeparator(DemucsConfig(device='cpu', model_repo=str(tmp_path)))
    with pytest.raises(ModelLoadingError):
        separator._load()


@pytest.mark.parametrize('kind', ['missing', 'file'])
def test_local_repo_requires_directory(tmp_path, kind):
    pytest.importorskip('demucs')
    repo = tmp_path / 'models'
    if kind == 'file':
        repo.write_text('not a directory')
    separator = DemucsSeparator(DemucsConfig(device='cpu', model_repo=str(repo)))
    with pytest.raises(FileNotFoundError, match='不是目录'):
        separator._load()


@pytest.mark.parametrize('model_name', ['custom', 'custom_bag'])
def test_custom_local_weights_load_without_shipped_filename(tmp_path, model_name):
    torch = pytest.importorskip('torch')
    pytest.importorskip('demucs')
    from demucs.htdemucs import HTDemucs

    original = HTDemucs(sources=['vocals', 'other'], channels=8, nfft=64,
                       depth=2, dconv_mode=0, t_layers=0)
    args, kwargs = original._init_args_kwargs
    torch.save({'klass': HTDemucs, 'args': args, 'kwargs': kwargs,
                'state': original.state_dict()}, tmp_path / 'custom.th')
    (tmp_path / 'custom_bag.yaml').write_text('models: [custom]\n')
    separator = DemucsSeparator(DemucsConfig(model_name=model_name, device='cpu',
                                             model_repo=str(tmp_path)))
    loaded = separator._load()
    model = loaded.models[0] if model_name == 'custom_bag' else loaded
    assert model.sources == original.sources
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
    assert separator._load() is loaded


def test_local_repo_retains_upstream_checksum_validation(tmp_path):
    pytest.importorskip('demucs')
    from demucs.repo import ModelLoadingError
    (tmp_path / 'custom-00000000.th').write_bytes(b'corrupt weights')
    separator = DemucsSeparator(DemucsConfig(model_name='custom', device='cpu',
                                             model_repo=str(tmp_path)))
    with pytest.raises(ModelLoadingError, match='checksum'):
        separator._load()


def test_local_repo_does_not_change_quality_parameters():
    from dataclasses import asdict
    plain = asdict(DemucsConfig(device='cpu'))
    local = asdict(DemucsConfig(device='cpu', model_repo='/relocated/models'))
    plain.pop('model_repo'); local.pop('model_repo')
    assert plain == local
