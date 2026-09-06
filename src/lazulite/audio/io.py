"""轻量音频文件元数据读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mutagen import File as mutagen_file

from lazulite.models.audio import AudioMetadata


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
