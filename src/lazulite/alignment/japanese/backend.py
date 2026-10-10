"""Local Wav2Vec2 CTC adapter with configuration-derived vocabulary and frames."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from lazulite.events import checkpoint
from .stages import FrameMap


def inspect_model(path):
    """Read small configuration files only; never instantiate or hash weights."""
    path = Path(path)
    config = json.loads((path/'config.json').read_text())
    if config.get('model_type') != 'wav2vec2' or config.get('add_adapter', False):
        raise ValueError('this adapter requires Wav2Vec2 CTC without output adapters')
    processor = path/'preprocessor_config.json'
    pre = json.loads(processor.read_text()) if processor.is_file() else json.loads((path/'processor_config.json').read_text())['feature_extractor']
    vocab = json.loads((path/'vocab.json').read_text())
    added = path/'added_tokens.json'
    if added.is_file(): vocab.update(json.loads(added.read_text()))
    stride, receptive = 1, 1
    kernels, strides = config['conv_kernel'], config['conv_stride']
    if not kernels or len(kernels) != len(strides) or any(type(v) is not int or v<=0 for v in kernels+strides):
        raise ValueError('invalid convolution layout')
    for kernel, step in zip(kernels,strides):
        receptive += (kernel-1)*stride
        stride *= step
    frames = FrameMap(pre['sampling_rate'],stride,receptive)
    columns, blank = config['vocab_size'], config['pad_token_id']
    if type(columns) is not int or columns<=0 or type(blank) is not int or not 0<=blank<columns or any(type(v) is not int or not 0<=v<columns for v in vocab.values()) or len(set(vocab.values()))!=len(vocab):
        raise ValueError('invalid model vocabulary/output layout')
    if not any((path/name).is_file() for name in ('model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json')):
        raise ValueError('model weights are missing')
    if pre.get('feature_size') != 1 or type(pre.get('do_normalize')) is not bool:
        raise ValueError('explicit mono feature extractor configuration required')
    return vocab, blank, columns, frames, pre['do_normalize']


def chunked_forward(y, forward, *, frames, vocabulary_size, normalize=True, cancel=None, event_sink=None):
    if y.ndim != 1 or y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError('expected finite float32 mono waveform')
    n = frames.count(len(y))
    if not n: return np.empty((0,vocabulary_size),np.float32)
    if normalize: y = (y-y.mean())/np.sqrt(y.var()+1e-7)
    result, covered = np.empty((n,vocabulary_size),np.float32), np.zeros(n,np.uint8)
    stride = frames.stride_samples
    core_frames = max(1,round(20*frames.sample_rate/stride))
    context_frames = max(1,round(2*frames.sample_rate/stride))
    total = (n+core_frames-1)//core_frames
    for i,start in enumerate(range(0,n,core_frames)):
        checkpoint(cancel,event_sink,'acoustic',i,total)
        stop = min(n,start+core_frames)
        lo = max(0,start-context_frames)*stride
        hi = min(len(y),(start+core_frames+context_frames)*stride)
        # Small receptive fields/strides and final cores still need their complete receptive window.
        hi = min(len(y),max(hi,(stop-1)*stride+frames.receptive_samples))
        lp = forward(y[lo:hi].copy())
        if lp.shape != (frames.count(hi-lo),vocabulary_size) or not np.isfinite(lp).all():
            raise ValueError('unexpected acoustic frame count/vocabulary/values')
        ix=np.arange(len(lp))+lo//stride
        keep=(ix>=start)&(ix<stop)
        result[ix[keep]]=lp[keep]; covered[ix[keep]]+=1
    if not (covered==1).all(): raise ValueError('acoustic cores must cover each frame exactly once')
    checkpoint(cancel,event_sink,'acoustic',total,total)
    return result


@dataclass(frozen=True, slots=True)
class NextFireConfig:
    model_path: Path
    device: str = 'cpu'


class NextFireBackend:
    def __init__(self, config: NextFireConfig):
        self.config=config
        self.vocab,self.blank,self.vocabulary_size,self.frames,self.normalize=inspect_model(config.model_path)
        self.special_tokens=tuple(k for k in self.vocab if len(k)!=1 or k=='|')
        self._model=None

    @property
    def description(self):
        return {'adapter':'wav2vec2-ctc','model_path':str(self.config.model_path),
                'device':self.config.device,'dtype':'float32','normalize':self.normalize}

    def emissions(self,audio,*,cancel=None,event_sink=None):
        if audio.sample_rate!=self.frames.sample_rate or audio.samples.shape[0]!=1:
            raise ValueError('acoustic input does not match configured sample rate/channels')
        if self.frames.count(audio.samples.shape[1])==0:
            return np.empty((0,self.vocabulary_size),np.float32)
        if self._model is None:
            import torch
            self._torch=torch
            from lazulite._runtime import preserve_random_state
            with preserve_random_state(torch):
                from transformers import Wav2Vec2ForCTC
                self._model=Wav2Vec2ForCTC.from_pretrained(self.config.model_path,local_files_only=True,
                    dtype=torch.float32).to(self.config.device).eval()
        def forward(y):
            t=self._torch
            from lazulite._runtime import preserve_random_state
            with preserve_random_state(t),t.inference_mode():
                return self._model(t.from_numpy(y).unsqueeze(0).to(self.config.device)).logits[0].log_softmax(-1).cpu().numpy()
        return chunked_forward(audio.samples[0],forward,frames=self.frames,vocabulary_size=self.vocabulary_size,
                               normalize=self.normalize,cancel=cancel,event_sink=event_sink)

    def close(self):
        self._model=None
