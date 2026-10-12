# Lazulite

面向本地音乐的 Python 歌词对齐计算库，提供命令行和可独立调用的阶段 API。

- 搜索网易云、QQ 音乐、酷狗与 LRCLIB 歌词候选。
- 使用 Demucs 分离完整人声，分析声学活动与人声片段。
- 行级对齐路线：提供通用的跨语言行级对齐能力，使用 Whisper 转写各人声片段，在歌词无参考时间的情况下使用动态规划将片段与歌词对齐，在歌词有参考时间的情况下估计歌词时间的总体/分段偏移
- 日语词级对齐路线：不依赖行级结果，使用人声片段拼接、Wav2Vec2 CTC 词级对齐、可选的行首提示校准和端点整理。
- 支持读取文本、LRC 和 TTML。

歌词搜索使用 RapidFuzz 的完整与局部字符串相似度，加权生成 0～100 的浮点元数据分数。评分保留原始精度，按现有文本规范化、繁简转换、字段权重和时长条件计算；最低分过滤与候选排序直接使用该分数。RapidFuzz 作为 Python 依赖由 pip 安装。

注意转写模型的输出只用于匹配，不会进入结果正文，当前可用的开源 ASR 模型都无法达到从头转写歌词正文的可用线，故 Lazulite 必须依赖外部输入的歌词文本。

## 安装

Python 3.12+，音频解码和重采样需要 FFmpeg：

```sh
python -m pip install .
python -m pip install -c constraints.txt '.[whisper,japanese,vocal]'
```

模型路径指向本地目录。依赖约束提供各个模块的独立依赖的安装版本参考，CUDA Torch 构建或安装应当按实际部署环境选择。

## 命令行

```sh
lazulite lyrics search song.m4a --output candidates.json
lazulite transcribe vocals.wav --backend whisper --model-path /models/whisper --device cpu --output transcript.json
lazulite align lyrics.ttml --vocals vocals.wav --backend whisper --model-path /models/whisper --device cpu --mode dp --output lines.json
lazulite align lyrics.ttml --audio song.m4a --backend whisper --model-path /models/whisper --separator-device cpu --device cpu --output lines.json
lazulite align-japanese --vocals vocals.wav --targets units.json --model-path /models/karaoke-ctc --output result.json
lazulite align-japanese --audio song.m4a --ttml lyrics.ttml --model-path /models/karaoke-ctc --output result.json
```

`--audio` 从原曲分离；`--vocals` 直接读取完整人声，音频零点对应原曲零点。TTML 日语投影支持正文与 Ruby，未知汉字读音保留为未解决目标。结果只写指定输出，不改写歌词或媒体。

显式跨进程交换阶段结果：

```sh
lazulite align-japanese --vocals vocals.wav --targets units.json --model-path /models/karaoke-ctc --acoustic-output acoustic.npz --decode-output decoded.json --output result.json
lazulite align-japanese --acoustic-input acoustic.npz --targets changed-readings.json --output changed.json
lazulite align-japanese --decode-input decoded.json --targets units.json --without-completion --output raw.json
lazulite align lyrics.ttml --vocals vocals.wav --backend whisper --model-path /models/whisper --device cpu --transcriptions-output transcription-stages.json --output lines.json
lazulite align lyrics.ttml --vocals vocals.wav --transcriptions-input transcription-stages.json --output lines-again.json
```

声学文件包含 NumPy 矩阵、词表、blank、原轴拼接区间及帧映射。解码文件保留读音目标、单元／行关联、首遍和采用路径。转写文件区分预选窗口与全窗口生成条件。结构错误或缺失阶段明确报错；文件由调用者显式指定。

## Python 阶段接口

```python
from pathlib import Path
from lazulite.audio.io import decode_native_audio
from lazulite.audio.preparation import prepare_existing_vocals
from lazulite.alignment.japanese.backend import NextFireBackend, NextFireConfig
from lazulite.alignment.japanese.run import compute_acoustics, decode_japanese, finalize_japanese

prepared = prepare_existing_vocals(decode_native_audio("vocals.wav"))
backend = NextFireBackend(NextFireConfig(Path("/models/karaoke-ctc")))
try:
    acoustic = compute_acoustics(prepared, backend=backend, include_activity=True)
finally:
    backend.close()
# request 是调用方提供的 JapaneseAlignmentRequest。
decoded = decode_japanese(acoustic, request)
result = finalize_japanese(decoded)
raw = finalize_japanese(decoded, completion=False)
```

已有分析传入 `analysis`；人工片段传入 `segments`，不会被自动分析覆盖。业务 ID 和分析描述可选。原混音 `reference` 使用原混音电平参考；未提供时使用满刻度参考，条件记入分析描述。

行级使用 `AlignmentPipeline(transcriber=backend).run_prepared(prepared, document)`；结果的 `transcriptions` 可交给 `AlignmentPipeline().run_prepared(..., transcriptions=...)`。单阶段入口为 `build_transcription_inputs`、`run_transcription_stage`、`align_lines`。

## 模型与时间契约

`NextFireBackend` 适配本地 Wav2Vec2 CTC：从配置读取采样率、卷积步长／感受野、输出列数、词表和 blank。支持同结构修改权重与相容配置；不同架构由调用方实现声学后端。`inspect_model(path)` 仅读小型配置和检查权重文件是否存在。

日语 Wav2Vec2 CTC 模型建议使用 [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn)，该模型针对日语歌唱后训练，需注意其使用 CC-BY-NC-SA-4.0 协议，禁止商用

日语 profile 将读音转换成 Hepburn 字母，再通过所选词表编码；不能表达的读音保留未解决状态。`decode_japanese(..., encoder=...)` 可显式提供不同目标编码。`FrameMap` 以样本为单位声明步长、感受野和偏移，端点使用半开帧区间，公开时间为秒。

假名转换和以下三项软代价直接使用了 [MiliKara](https://github.com/BilyHurington/MiliKara) 的代码，并按 Lazulite 的阶段接口与时间轴映射作了适配。原作者版权和 MIT 许可声明保留于 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)。

`JapaneseConfig` 默认启用外部行首提示校准和局部端点补全；三个软代价默认关闭，可独立开启，作用重叠时相加。设一帧时长为 `Δt` 秒，该帧休止强度为 `rest`，从 CTC 路径分数中扣除：

| 开关 | 作用位置 | 每帧扣分 |
| --- | --- | --- |
| `line_blank` | 同一歌词行相邻目标 token 之间的 CTC blank 状态 | `Δt` |
| `rest_blank` | 同一歌词行相邻目标 token 之间的 CTC blank 状态 | `4 × rest × Δt` |
| `rest_token` | 所有目标 token 状态 | `25 × rest × Δt` |

`rest ∈ [0, 1]` 是相对人声能量推导的休止强度，越大表示越接近休止。它从原轴人声的 30 ms RMS／10 ms 步长、300 ms 功率平滑计算，再积分到实际声学帧，不跨删段接缝平滑；参考电平不可用时将 `rest` 置零。有限样本的消融结果显示这些代价可能并不会显著提高对齐精度，部分条件下甚至会降低效果，使用前需自行评估。

结果保留每个目标 ID、原始／建议端点、跨删段 fragments 和问题。`null` 表示没有新建议；`complete` 表示覆盖完整，不保证准确。Whisper 的概率也不代表时间正确率。

CTC 默认额外内存预算 512 MiB；超限抛出 `MemoryError`。取消在阶段、声学块、转写窗口和 CTC 帧边界检查，抛出 `events.Cancelled`。调用方控制已返回结果与模型生命周期。

## 开发

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

接口细节见[架构说明](docs/architecture.md)，测试与自备素材评估见[测试说明](tests/README.md)。来源和许可见 `THIRD_PARTY_LICENSES`。
