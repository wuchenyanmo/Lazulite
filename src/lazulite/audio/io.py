"""轻量音频文件元数据读取。"""

from __future__ import annotations

import hashlib
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(channels),
        "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
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
    return AudioBuffer(samples=samples, sample_rate=sample_rate, source_id=f"sha256:{_file_sha256(path)}")
