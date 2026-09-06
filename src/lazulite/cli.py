"""命令行边界。

此模块刻意不导入音频分离或转写后端；具体命令在相应能力完成后再装配。
"""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """构造不触发模型加载的基础命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="lazulite",
        description="歌词搜索、转写证据与行级对齐自动化引擎",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行轻量 CLI；后续子命令在此边界安装运行资源。"""
    build_parser().parse_args(argv)
    return 0
