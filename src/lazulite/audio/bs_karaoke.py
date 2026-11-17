"""Local BS Karaoke lead-vocal separation with bounded overlap-add inference."""
from dataclasses import dataclass
from pathlib import Path

from lazulite.models.audio import AudioBuffer
from .io import resample_audio


@dataclass(frozen=True, slots=True)
class BSKaraokeConfig:
    model_repo: str
    device: str = 'cpu'
    weights: str = 'bs_roformer_karaoke_frazer_becruily.ckpt'
    configuration: str = 'config_karaoke_frazer_becruily.yaml'


class BSKaraokeSeparator:
    def __init__(self, config: BSKaraokeConfig):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        import yaml
        from ._bs_roformer.bs_roformer import BSRoformer
        from lazulite._runtime import preserve_random_state
        # The published configuration uses tuple tags; support only this extra
        # data type, without enabling Python object construction.
        class ConfigLoader(yaml.SafeLoader):
            pass
        ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                     lambda loader, node: tuple(loader.construct_sequence(node)))
        root = Path(self.config.model_repo)
        self._settings = yaml.load((root / self.config.configuration).read_text(), Loader=ConfigLoader)
        if self._settings['audio']['sample_rate'] != 44100:
            raise ValueError('BS Karaoke requires a 44100 Hz configuration')
        size = self._settings['audio']['chunk_size']
        overlap = self._settings['inference']['num_overlap']
        if not isinstance(size, int) or not isinstance(overlap, int) or not 1 <= overlap <= size or size < 10:
            raise ValueError('Invalid BS Karaoke window configuration')
        with preserve_random_state(torch):
            model = BSRoformer(**self._settings['model'])
        model.load_state_dict(torch.load(root / self.config.weights, map_location='cpu', weights_only=True), strict=True)
        self._model = model.to(self.config.device).eval()
        return self._model

    def separate(self, audio: AudioBuffer) -> AudioBuffer:
        import numpy as np
        import torch
        import torch.nn.functional as F
        prepared = resample_audio(audio, sample_rate=44100, channels=2)
        model = self._load()
        # Accumulation stays on CPU; only a single window resides on the device.
        mix = torch.from_numpy(prepared.samples.copy())
        size = self._settings['audio']['chunk_size']
        step = size // self._settings['inference']['num_overlap']
        border = size - step
        padded = border > 0 and mix.shape[-1] > 2 * border
        if padded:
            mix = F.pad(mix, (border, border), mode='reflect')
        fade = size // 10
        window = torch.ones(size)
        window[:fade] = torch.linspace(0, 1, fade)
        window[-fade:] = torch.linspace(1, 0, fade)
        result, counter = torch.zeros_like(mix), torch.zeros_like(mix)
        with torch.inference_mode():
            for offset in range(0, mix.shape[-1], step):
                part = mix[:, offset:offset + size]
                length = part.shape[-1]
                if length < size:
                    part = F.pad(part, (0, size - length), mode='reflect' if length > size // 2 + 1 else 'constant')
                output = model(part.unsqueeze(0).to(self.config.device))[0]
                if output.ndim == 3:
                    output = output[0]
                output = output.detach().float().cpu()
                weights = window.clone()
                if offset == 0:
                    weights[:fade] = 1
                if offset + size >= mix.shape[-1]:
                    weights[-fade:] = 1
                result[:, offset:offset + length] += output[:, :length] * weights[:length]
                counter[:, offset:offset + length] += weights[:length]
        output = (result / counter).numpy()
        if padded:
            output = output[:, border:-border]
        if not np.isfinite(output).all():
            raise RuntimeError('BS Karaoke produced nonfinite samples')
        return AudioBuffer(output, 44100, audio.source_id)

    def close(self):
        self._model = None
