"""Pinned NextFire FP32 frontend, matching research 20s cores ±2s context."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from lazulite.events import checkpoint

REVISION = '2ab2b5f46539ee284703c281f286b01d2410ee12'
VERSION = 'nextfire-whole-normalized-old-context2-v1'
VOCAB = {c:i+1 for i,c in enumerate('abcdefghijklmnopqrstuvwxyz')} | {'|':0,"'":27,'[UNK]':28,'[PAD]':29}
HASHES = {
 'model.safetensors':'4d1d6fbb543534ab32bcceac5f5a42052c7fde3d16a83cbbce858473d5735042',
 'config.json':'665dbd5be3a170d0b53f86e702a4f65f755471a2d81c1e0cf35ba566a72bf382',
 'processor_config.json':'8bc3c8d6d97ea61524d5c15d8554b4baf072d2264c0bda1839bc7a822c200d66',
 'vocab.json':'26dc928520b2bde28e1c9e7fd425aae6b4eb40cd992bcb0bcc16ca6fada46491',
}


def chunked_forward(y, forward, *, cancel=None, event_sink=None):
    if y.ndim != 1 or y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError('expected finite float32 mono 16kHz waveform')
    n = max(0,(len(y)-400)//320+1)
    if n == 0:
        return np.empty((0,32),np.float32)
    # Exactly the Wav2Vec2FeatureExtractor whole-waveform normalization.
    y = (y-y.mean())/np.sqrt(y.var()+1e-7)
    result, covered = np.empty((n,32),np.float32), np.zeros(n,np.uint8)
    total = (len(y)+319999)//320000
    for i,core in enumerate(range(0,len(y),320000)):
        checkpoint(cancel,event_sink,'acoustic',i,total)
        lo,hi = max(0,core-32000), min(len(y),core+352000)
        lp = forward(y[lo:hi].copy())
        if lp.shape != ((hi-lo-400)//320+1,32) or not np.isfinite(lp).all():
            raise ValueError('unexpected NextFire frame count/vocabulary/values')
        ix = np.arange(len(lp))+lo//320
        keep = (ix>=core//320)&(ix<min((core+320000)//320,n))
        result[ix[keep]]=lp[keep]; covered[ix[keep]]+=1
    if not (covered == 1).all():
        raise ValueError('acoustic cores must cover each frame exactly once')
    checkpoint(cancel,event_sink,'acoustic',total,total)
    return result


@dataclass(frozen=True, slots=True)
class NextFireConfig:
    model_path: Path
    device: str = 'cpu'
    revision: str = REVISION

    def __post_init__(self):
        if self.revision != REVISION:
            raise ValueError('unsupported NextFire revision')


class NextFireBackend:
    vocab = VOCAB
    blank = 29

    def __init__(self, config: NextFireConfig):
        self.config = config
        self._model = None
        self._verified = False

    def _verify(self):
        if self._verified:
            return
        for name, expected in HASHES.items():
            with (Path(self.config.model_path)/name).open('rb') as f:
                actual = hashlib.file_digest(f,'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'NextFire content identity mismatch: {name}')
        self._verified = True

    @property
    def cache_identity(self):
        self._verify()
        from importlib.metadata import version
        return {'revision':REVISION,'files':HASHES,'algorithm':VERSION,'dtype':'float32',
                'device':self.config.device,'torch':version('torch'),'transformers':version('transformers')}

    def emissions(self, audio, *, cancel=None, event_sink=None):
        if audio.sample_rate != 16000 or audio.samples.shape[0] != 1:
            raise ValueError('NextFire requires mono 16 kHz')
        self._verify()
        if audio.samples.shape[1] < 400:
            return np.empty((0,32),np.float32)
        if self._model is None:
            import torch
            from transformers import Wav2Vec2ForCTC
            self._torch = torch
            self._model = Wav2Vec2ForCTC.from_pretrained(self.config.model_path, local_files_only=True,
                                                       torch_dtype=torch.float32).to(self.config.device).eval()
        def forward(y):
            t = self._torch
            with t.inference_mode():
                return self._model(t.from_numpy(y).unsqueeze(0).to(self.config.device)).logits[0].log_softmax(-1).cpu().numpy()
        return chunked_forward(audio.samples[0],forward,cancel=cancel,event_sink=event_sink)

    def close(self):
        self._model = None
