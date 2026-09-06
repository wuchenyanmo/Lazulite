# Lazulite

Lazulite 是一个可安装的歌词自动化引擎：它读取音频元数据、搜索并重排歌词候选，分离人声并生成声学窗口，通过可替换 ASR 后端建立转写证据，再输出行级时间候选。它不保存项目状态、不覆盖输入媒体，也不提供 TTML 编辑器或 Web 服务。

## 安装

基础功能不安装 Torch、Demucs 或模型。需要 Python 3.12 及 FFmpeg：

```bash
python -m pip install .
```

按实际能力安装 extra；CUDA Torch wheel 的来源由部署环境决定。Qwen 与 Whisper 使用已验证的约束版本：

```bash
python -m pip install --constraint constraints.txt '.[qwen,whisper,vocal,forced-qwen]'
```

模型必须由调用方下载并指定本地目录。每次命令均要求填写固定 Hub revision 或本地权重 SHA256；这会进入 JSON 证据和缓存身份，不能用不明确的 `local` 占位。

## 命令行

```bash
lazulite lyrics search song.m4a --output candidates.json

lazulite transcribe song.m4a \
  --backend whisper --model-path /models/whisper-large-v3 \
  --revision <固定revision> --language ja --output transcript.json

lazulite align song.m4a lyrics.ttml \
  --backend qwen --model-path /models/Qwen3-ASR-1.7B \
  --revision <固定revision或权重SHA256> --mode auto --output alignment.json

lazulite forced-align clip.wav confirmed.txt \
  --backend qwen --model-path /models/Qwen3-ForcedAligner-0.6B \
  --revision <固定revision或权重SHA256> --language ja --output fine.json
```

`align` 默认使用 Demucs；仅在已经输入分离人声或进行诊断时使用 `--without-separation`。`--full-distributions DIR` 可按片段导出完整 `logprob` NPZ；普通 JSON 只保留生成步、选中 token 概率和 top-k 摘要。Qwen 没有 token 时间，因此其行级粗时间来自声学 core 窗口，不会被误标为模型时间。

## Python API

```python
from lazulite.audio.io import read_audio_metadata
from lazulite.lyrics import fetch_lyrics, rank_lyrics, search_lyrics
from lazulite.audio.activity import analyze_vocals
from lazulite.transcription.run import transcribe_segments

metadata = read_audio_metadata("song.m4a")
candidates = search_lyrics(metadata)
lyrics = fetch_lyrics(candidates)
analysis = analyze_vocals(audio)
transcription = transcribe_segments(audio, analysis.segments, transcriber=backend)
ranked = rank_lyrics(lyrics, transcription, analysis)
```

`AlignmentPipeline` 只组合已注入的 separator、转写器、可选 `FileCache` 和事件回调。缓存仅在音频 `source_id` 是 SHA256 且运行资源提供完整模型/参数身份时启用；其键包含输入内容、算法版本、歌词投影、转写配置和模型 revision。缓存目录必须由调用方传入，库不会创建用户项目或自动清理目录。

## 时间与 forced alignment 限制

所有公开时间均为相对于整曲起点的秒数。未匹配的歌词行保留在结果中，时间为 `null`；多行共用声学窗口时只标 `interpolated`。`confidence=None` 与合法的 `0` 不同，缺失置信度不会自动丢弃候选。

Qwen forced aligner 是独立实验后端，只对调用方已确认的一段正文工作。它按实际返回的字符或词输出范围；无法明确回写的单位会记录诊断。它不承诺日语 mora/phoneme 对齐，也不把人工 TTML 时间作为自动对齐输入。

## 开发验证与基准

```bash
python -m pytest
python -m build
python tools/benchmark.py --backend qwen --model-path /models/Qwen3-ASR-0.6B --revision <revision> --output benchmark.json
```

`tests/fixtures` 是三首整曲基准素材，不随 wheel 安装，也不在每次 pytest 中运行模型。基准方法、指标和最近一次记录见 [docs/benchmark.md](docs/benchmark.md)。重构范围和模型字段实测依据见 [docs/refactoring.md](docs/refactoring.md)。
