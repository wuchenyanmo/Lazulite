"""Demucs 人声分离器，模型状态仅属于实例。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

from lazulite.models.audio import AudioBuffer

from .io import resample_audio


@dataclass(frozen=True, slots=True)
class DemucsConfig:
    model_name: str = "htdemucs"
    device: str = "cuda"
    model_revision: str | None = None
    split: bool = True
    overlap: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < 1:
            raise ValueError("Demucs overlap 必须在 [0, 1) 内")


class DemucsSeparator:
    def __init__(self, config: DemucsConfig = DemucsConfig()) -> None:
        self._config = config; self._model = None; self._torch = None

    def _load(self):
        if self._model is not None: return self._model
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError as error:
            raise RuntimeError("人声分离需要安装 lazulite[vocal]") from error
        self._torch = torch; self._apply_model = apply_model
        self._model = get_model(self._config.model_name).to(self._config.device).eval()
        return self._model

    @property
    def cache_identity(self) -> dict[str, object] | None:
        """未指定权重身份时不允许缓存分离后的自动化结果。"""
        if not self._config.model_revision:
            return None
        def installed(package: str) -> str:
            try:
                return version(package)
            except PackageNotFoundError:
                return "unavailable"
        return {
            "separator": "demucs",
            "runtime_version": f"demucs {installed('demucs')}; torch {installed('torch')}",
            "config": asdict(self._config),
        }

    def separate(self, audio: AudioBuffer) -> AudioBuffer:
        prepared = resample_audio(audio, sample_rate=44_100, channels=2)
        model = self._load(); waveform = self._torch.from_numpy(prepared.samples).unsqueeze(0).to(self._config.device)
        with self._torch.inference_mode(): sources = self._apply_model(model, waveform, device=self._config.device, progress=False, split=self._config.split, overlap=self._config.overlap)
        vocals = sources[0, model.sources.index("vocals")].detach().float().cpu().numpy()
        return AudioBuffer(vocals, 44_100, audio.source_id)

    def close(self) -> None:
        # 由调用方决定是否清空 CUDA allocator，库仅结束本实例的模型所有权。
        self._model = None
