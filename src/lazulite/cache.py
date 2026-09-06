"""可选的单目录结构化文件缓存。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import warnings
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    """对确定性 JSON 计算 SHA256，调用方负责放入模型与算法身份。"""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class FileCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get_json(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{key}.json"
        try:
            return json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            warnings.warn(f"缓存 JSON 已损坏，忽略本次命中: {path}", RuntimeWarning)
            return None

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{key}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False) as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            temporary = Path(handle.name)
        temporary.replace(target)
