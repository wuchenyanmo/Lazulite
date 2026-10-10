"""轻量音频文件元数据读取。"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from mutagen import File as mutagen_file

from lazulite.models.audio import AudioBuffer, AudioMetadata


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace").strip()
        return text or None
    if isinstance(value, (list, tuple)):
        for item in value:
            if text := _first_text(item):
                return text
        return None
    if hasattr(value, "text"):
        return _first_text(value.text)
    text = str(value).strip()
    return text or None


def read_audio_metadata(audio_path: str | Path) -> AudioMetadata:
    """读取可搜索的标题、艺人、专辑和时长；不修改源媒体。"""
    path = Path(audio_path).expanduser().resolve()
    audio = mutagen_file(path, easy=False)
    if audio is None or getattr(audio, "info", None) is None:
        raise ValueError(f"无法读取音频文件或不支持的格式: {path}")
    tags = getattr(audio, "tags", None) or {}
    suffix = path.suffix.lower()
    if suffix in {".m4a", ".mp4", ".aac", ".alac"}:
        title = _first_text(tags.get("\xa9nam"))
        artist = _first_text(tags.get("\xa9ART")) or _first_text(tags.get("aART"))
        album = _first_text(tags.get("\xa9alb"))
    elif suffix == ".mp3":
        title = _first_text(tags.get("TIT2"))
        artist = _first_text(tags.get("TPE1")) or _first_text(tags.get("TPE2"))
        album = _first_text(tags.get("TALB"))
    else:
        title = _first_text(tags.get("title"))
        artist = _first_text(tags.get("artist")) or _first_text(tags.get("albumartist"))
        album = _first_text(tags.get("album"))
    return AudioMetadata(
        path=path,
        title=title or path.stem,
        artist=artist,
        album=album,
        duration=float(audio.info.length),
    )


def decode_audio(
    audio_path: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> AudioBuffer:
    """用 FFmpeg 解码并重采样；样本零点始终保留为整曲时间零点。"""
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate 与 channels 必须为正数")
    path = Path(audio_path).expanduser().resolve()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("未找到 FFmpeg，无法解码音频") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"无法解码音频 {path}: {detail}") from error
    values = np.frombuffer(completed.stdout, dtype="<f4")
    if values.size == 0 or values.size % channels:
        raise ValueError(f"FFmpeg 输出了无效的 PCM 数据: {path}")
    samples = values.reshape(-1, channels).T.copy()
    return AudioBuffer(
        samples=samples,
        sample_rate=sample_rate,
        source_id=str(path),
    )


def resample_audio(
    audio: AudioBuffer, *, sample_rate: int, channels: int | None = None
) -> AudioBuffer:
    """对内存音频做FFmpeg 带限重采样；模型输入的布局保持显式。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须为正数")
    target_channels = channels if channels is not None else audio.samples.shape[0]
    if target_channels <= 0:
        raise ValueError("channels 必须为正数")
    source = audio.samples
    if target_channels == 1:
        source = source.mean(axis=0, keepdims=True)
    elif target_channels == 2 and source.shape[0] == 1:
        source = np.repeat(source, 2, axis=0)
    elif target_channels != source.shape[0]:
        raise ValueError("不支持该声道数转换")
    if sample_rate == audio.sample_rate:
        return AudioBuffer(
            source.astype(np.float32, copy=True), sample_rate, audio.source_id
        )
    if source.shape[1] == 0:
        return AudioBuffer(
            source.astype(np.float32, copy=True), sample_rate, audio.source_id
        )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(audio.sample_rate),
        "-ac",
        str(target_channels),
        "-i",
        "pipe:0",
        "-af",
        f"aresample={sample_rate}:resampler=swr:filter_size=32:phase_shift=10:cutoff=0.97",
        "-f",
        "f32le",
        "pipe:1",
    ]
    # Flush the finite filter even for clips shorter than its delay. Preserve sample
    # zero and return ceil(duration * rate), matching the duration contract.
    target_length = math.ceil(source.shape[1] * sample_rate / audio.sample_rate)
    padding = math.ceil(64 * max(1, audio.sample_rate / sample_rate))
    padded = np.pad(source, ((0, 0), (0, padding)))
    try:
        output = subprocess.run(
            command,
            input=padded.T.astype("<f4").tobytes(),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("未找到 FFmpeg，无法重采样") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(error.stderr.decode("utf-8", errors="replace")) from error
    samples = (
        np.frombuffer(output.stdout, dtype="<f4")
        .reshape(-1, target_channels)
        .T[:, :target_length]
        .copy()
    )
    if samples.shape[1] != target_length:
        raise ValueError("FFmpeg 返回了不完整的重采样音频")
    return AudioBuffer(samples, sample_rate, audio.source_id)


def decode_native_audio(audio_path: str | Path) -> AudioBuffer:
    """Explicit stem import preserves the file's decoded sample rate and channels."""
    import json
    path=Path(audio_path).expanduser().resolve()
    try:
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
                              'stream=sample_rate,channels','-of','json',str(path)],check=True,capture_output=True)
        streams=json.loads(probe.stdout)['streams']
        if len(streams)!=1:raise ValueError('one selected audio stream required')
        rate,channels=int(streams[0]['sample_rate']),int(streams[0]['channels'])
    except FileNotFoundError as error:
        raise RuntimeError('未找到 FFprobe，无法确认人声文件布局') from error
    except (subprocess.CalledProcessError,KeyError,ValueError,TypeError) as error:
        raise ValueError('无法确认人声文件采样率和声道') from error
    return decode_audio(path,sample_rate=rate,channels=channels)
