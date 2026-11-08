"""Local Wav2Vec2 CTC adapter with configuration-derived vocabulary and frames."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from lazulite.events import checkpoint
from .stages import FrameMap


def _object_file(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def _shards(path):
    mapping = _object_file(path).get('weight_map')
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(v,str) for v in mapping.values()):
        raise ValueError('invalid sharded weight index')
    return set(mapping.values())


class UnsupportedModel(ValueError):
    """Directory is not one of the supported acoustic formats."""


def inspect_hf_model(path):
    """Read small configuration files only; never instantiate or hash weights."""
    path = Path(path)
    config = _object_file(path/'config.json')
    if config.get('model_type') != 'wav2vec2' or config.get('add_adapter', False):
        raise UnsupportedModel('this adapter requires Wav2Vec2 CTC without output adapters')
    processor = path/'preprocessor_config.json'
    pre = _object_file(processor) if processor.is_file() else _object_file(path/'processor_config.json')['feature_extractor']
    if not isinstance(pre, dict): raise ValueError('feature extractor must be an object')
    vocab = _object_file(path/'vocab.json')
    added = path/'added_tokens.json'
    if added.is_file(): vocab.update(_object_file(added))
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
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path/name).is_file():
            for shard in _shards(path/name):
                contained_file(path, shard)
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


MMS_VOCAB = {c: i for i, c in enumerate("-aienoutsrmkldghybpwcvjzf'q x".replace(" ", ""))}


def contained_file(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('weight path must be relative to the model directory')
    root = Path(root).resolve()
    file = (root / name).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise ValueError('weight file is missing or outside the model directory')
    return file


def describe_model(path):
    """Inspect one supported model directory without loading weights or devices."""
    path = Path(path).resolve()
    declaration = path / 'alignment-model.json'
    if declaration.is_file():
        value = _object_file(declaration)
        if value.get('type') != 'torchaudio-mms-fa':
            raise UnsupportedModel('unsupported alignment model type')
        weights = contained_file(path, value.get('weights'))
        return {'type': 'torchaudio-mms-fa', 'vocabulary': MMS_VOCAB.copy(), 'blank': 0,
                'columns': 28, 'frames': FrameMap(16000, 320, 400), 'normalize': True,
                'files': (declaration, weights)}
    vocab, blank, columns, frames, normalize = inspect_hf_model(path)
    names = ('config.json', 'preprocessor_config.json', 'processor_config.json', 'vocab.json',
             'added_tokens.json', 'model.safetensors', 'pytorch_model.bin',
             'model.safetensors.index.json', 'pytorch_model.bin.index.json')
    files = [path/name for name in names if (path/name).is_file()]
    for file in tuple(files):
        if file.name.endswith('.index.json'):
            files.extend(contained_file(path, name) for name in _shards(file))
    return {'type': 'wav2vec2-ctc', 'vocabulary': vocab, 'blank': blank, 'columns': columns,
            'frames': frames, 'normalize': normalize, 'files': tuple(files)}


def inspect_model(path):
    info = describe_model(path)
    return info['vocabulary'], info['blank'], info['columns'], info['frames'], info['normalize']


class MMSFABackend(NextFireBackend):
    @property
    def description(self):
        return {**super().description, 'adapter': 'torchaudio-mms-fa'}

    def emissions(self, audio, *, cancel=None, event_sink=None):
        if audio.sample_rate != self.frames.sample_rate or audio.samples.shape[0] != 1:
            raise ValueError('acoustic input does not match configured sample rate/channels')
        checkpoint(cancel)
        if not self.frames.count(audio.samples.shape[1]):
            return np.empty((0, 28), np.float32)
        if self._model is None:
            import torch
            from torchaudio.models import wav2vec2_model
            from lazulite._runtime import preserve_random_state
            with preserve_random_state(torch):
                model = wav2vec2_model(
                    extractor_mode='layer_norm', extractor_conv_layer_config=[(512,10,5)]+[(512,3,2)]*4+[(512,2,2)]*2,
                    extractor_conv_bias=True, encoder_embed_dim=1024, encoder_projection_dropout=0.0,
                    encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16, encoder_num_layers=24,
                    encoder_num_heads=16, encoder_attention_dropout=0.0, encoder_ff_interm_features=4096,
                    encoder_ff_interm_dropout=0.1, encoder_dropout=0.0, encoder_layer_norm_first=True,
                    encoder_layer_drop=0.1, aux_num_out=28)
                state = torch.load(describe_model(self.config.model_path)['files'][1], map_location='cpu', weights_only=True)
                # Original MMS_FA checkpoint reserves rows 1, 2, 3 for unused symbols.
                if tuple(state['aux.weight'].shape) != (31, 1024) or tuple(state['aux.bias'].shape) != (31,):
                    raise ValueError('MMS_FA requires the original 31-row output head')
                keep = [0] + list(range(4, 31))
                state['aux.weight'] = state['aux.weight'][keep]
                state['aux.bias'] = state['aux.bias'][keep]
                model.load_state_dict(state, strict=True)
                self._model = model.to(self.config.device).eval()
            self._torch = torch
        def forward(y):
            t = self._torch
            with t.inference_mode():
                logits, _ = self._model(t.from_numpy(y).unsqueeze(0).to(self.config.device))
                return logits[0].log_softmax(-1).cpu().numpy()
        return chunked_forward(audio.samples[0], forward, frames=self.frames, vocabulary_size=28,
                               normalize=True, cancel=cancel, event_sink=event_sink)


def open_backend(path, device='cpu'):
    info = describe_model(path)
    cls = MMSFABackend if info['type'] == 'torchaudio-mms-fa' else NextFireBackend
    return cls(NextFireConfig(Path(path), device))
