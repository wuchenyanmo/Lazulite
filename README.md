# Lazulite

面向本地音乐的 Python 歌词自动化引擎，提供命令行和可独立调用的 API。

- 从网易云、QQ 音乐、酷狗和 LRCLIB 搜索歌词，并按元数据与转写内容排序。
- 通过 Demucs 分离人声、分析声学活动并生成转写窗口。
- 使用 Whisper 或 Qwen3-ASR（0.6B / 1.7B）转写，输出 token 概率、top-k 和可选完整分布。
- 读取 LRC、文本与 TTML 歌词投影，生成行级时间候选及来源诊断。
- 使用实验性 Qwen forced aligner，将确认文本映射到字符或词级时间。

## 安装

使用 Python 3.12+；音频解码需要 FFmpeg。在仓库目录运行：

```bash
python -m pip install .
```

按需安装模型能力，例如：

```bash
python -m pip install -c constraints.txt '.[whisper,qwen,vocal,forced-qwen]'
```

模型命令使用本地权重目录和对应的固定 revision 或权重 SHA256。CUDA Torch 安装来源按部署环境选择。

## 命令行

将示例中的模型路径和 `REVISION` 替换为实际值：

```bash
lazulite lyrics search song.m4a --output candidates.json

lazulite transcribe song.m4a \
  --backend whisper --model-path /models/whisper-large-v3 \
  --revision REVISION --language ja --output transcript.json

lazulite align song.m4a lyrics.ttml \
  --backend qwen --model-path /models/Qwen3-ASR-1.7B \
  --revision REVISION --language ja --mode dp --output alignment.json

lazulite forced-align clip.wav confirmed.txt \
  --backend qwen --model-path /models/Qwen3-ForcedAligner-0.6B \
  --revision REVISION --language ja --output fine.json
```

`align` 默认使用 Demucs，已分离的人声可加 `--without-separation`。转写和行对齐命令可加 `--full-distributions DIR` 导出完整 logprob NPZ。公开时间采用整曲绝对秒，结果标注模型时间、声学窗口或插值来源，供后续人工复核。

## Python API

各阶段可单独调用，例如搜索和获取歌词：

```python
from lazulite.audio.io import read_audio_metadata
from lazulite.lyrics import fetch_lyrics, search_lyrics

candidates = search_lyrics(read_audio_metadata("song.m4a"))
lyrics = fetch_lyrics(candidates)
```

`transcribe_segments` 接受实现 `Transcriber` 协议的后端；`align_lines` 可复用已有转写和声学结果。`AlignmentPipeline` 组合分离、转写与行对齐，支持事件回调和 `FileCache(root)`。缓存目录由调用方指定，按音频内容、模型身份和有效配置复用结果。模型实例通过 `close()` 结束生命周期。

## 开发

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

整曲评估方法和结果见 [基准记录](docs/benchmark.md)，接口约定与模型实测入口见 [重构说明](docs/refactoring.md)。
