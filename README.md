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

### 共享人声准备

`audio.preparation.prepare_vocals(original, separator=...)` 从原曲分离并分析。
`prepare_existing_vocals(stem, original_id=..., original_duration=...)` 显式分析完整人声；
传入已有 `analysis` 时必须同时提供 `analysis_identity`。人工片段通过 `segments` 提供，保留自动分析原结果。
`PreparedVocals` 保存实际 PCM SHA256、零点、配置及片段来源；`write_prepared` / `read_prepared`
使用原始 float32 NumPy 数组和 `lazulite-prepared-v1` manifest，无量化或增益处理。
行级 `AlignmentPipeline.run_prepared(prepared, document)` 复用准备产物。

```sh
lazulite prepare-vocals song.m4a --output-dir prepared --separator-device cpu
lazulite prepare-vocals --vocals vocals.wav --original-id sha256:ORIGINAL_HASH --original-duration 180 --output-dir prepared
```

Demucs 显式记录 shifts=1、seed=0、split、overlap；分离后恢复调用方随机状态。
这是一项新的可复现输入条件，不能与旧随机分离缓存混同。

### 独立日语全局 CTC

安装 `lazulite[japanese]`；原曲分离另需 `vocal`。NextFire 权重固定为
`2ab2b5f46539ee284703c281f286b01d2410ee12`，本地文件逐项校验 SHA256。
`models.japanese.JapaneseAlignmentRequest` 明确传入正文 code point 范围、读音、语言及来源。
`alignment.japanese.run.align_japanese` 不构造 Whisper，不消费行级 DP。
空读音保留 ID；`null` 端点代表没有新建议，不代表删除旧值。

```sh
lazulite align-japanese --prepared prepared/manifest.json --targets units.json --model-path /path/to/NextFire --device cpu --hint-mode off --output result.json
```

结果 schema 为 `lazulite-japanese-result-v1`：原始路径、建议端点、原轴 fragments、拼接区间、问题和覆盖分别保存。
CTC 默认回溯预算 512 MiB，滚动 float64 分数＋uint8 回溯；超限明确失败。
声学块间与每 128 帧解码检查取消回调，抛出 `events.Cancelled`。
固定 profile/假名计算的 MIT 来源及许可见 `THIRD_PARTY_LICENSES`。

### 输入、状态与证据

日语默认 `hint_mode="auto"`，只在外部行首通过投票、完整请求前后覆盖和漂移门控后执行第二遍；
否则返回首遍。提示在删除间隔内不吸附。默认启用局部补全；`--without-completion` 可单独观察基线。
TTML 可用 `--ttml lyrics.ttml` 替代 `--targets units.json`，支持无行时间、Ruby 和秒/ms/时钟值；
翻译、音译、背景不进入正文。不支持 tick/frame 或带时间的父容器，明确报错。
未知汉字保持缺读音；快照生成的单元 ID 不保证跨文本修改稳定。

行级结果提供 `status`、`coverage`、`issue_counts` 和起止各自来源；`offset_accepted` 只用于 offset 接受决策，
不是泛化准确率。`reliable` 已移除，旧缓存直接失效。Whisper 每个实际窗口有独立 ID、可空 core 交集，
原始生成时间、Unicode/BPE 步映射、停止原因保存在证据中，派生 end 不改写原始数组。
`--max-new-tokens` 默认 128，可显式设 256。`--evidence-policy generic|whisper-lexical-v1` 控制统一风险公式；
缺失概率摘要不受 Whisper 阈值惩罚。

```sh
lazulite align lyrics.ttml --prepared prepared/manifest.json --backend whisper --model-path /path/to/whisper-large-v3 --revision MODEL_REVISION --device cpu --mode dp --max-new-tokens 128 --output lines.json
```

`FileCache` 按人声准备、实际转写请求、日语声学概率和原始解码路径分别复用。数组先写完整内容再原子发布 manifest，
坏文件不能命中。改 DP 配置可复用转写，改读音可复用日语声学概率。外部后端没有可信 `cache_identity` 时禁用模型产物缓存。
`AlignmentPipeline(..., cancel=callback)` 与 `align_japanese(..., cancel=callback)` 的取消回调返回 true 时抛出 `Cancelled`；
模型内部单次前向不抢占，块/请求间检查；取消不会发布最终成功结果。
