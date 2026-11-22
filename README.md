# Lazulite

面向本地音乐的歌词搜索、人声分离与自动对齐 Python 库，提供命令行工具和可按阶段调用的 Python 接口。

配套的桌面编辑与打轴工具见 [Lazulite Studio](https://github.com/zet-wu/Lazulite-studio)。

## 主要功能

- **在线歌词搜索与读取**：支持从网易云音乐、QQ 音乐、酷狗与 LRCLIB 搜索歌词候选，并支持读取纯文本、LRC 和 TTML 歌词。
- **人声分离与声学分析**：默认使用 [Demucs](https://github.com/adefossez/demucs) 分离人声，也可选用 [BS Karaoke](https://huggingface.co/becruily/bs-roformer-karaoke)（能更好地剔除和声与背景人声，但耗时更长，不建议仅靠 CPU 推理）。分离后自动划分人声片段，为后续对齐提供区间。
- **CTC 词级对齐**：结合人声片段与 Wav2Vec2 CTC 模型，完成细粒度的词/读音级时间轴对齐。支持日语专用模型 [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) 与跨语言通用模型 [MMS_FA](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html)，并支持假名转罗马字、基于 uroman 的拉丁归一化及声调数字剥离，可适配日语及多种语言。
- **通用行级对齐**：使用 Whisper 对人声片段进行转写辅助，在歌词缺少时间时通过动态规划匹配各行区间，在歌词已有参考时间时估计整体或分段偏移。转写结果仅用于定位匹配，不会改写用户输入的歌词正文。

## 安装

需要 Python 3.12 及以上版本，以及可在 `PATH` 中调用的 `ffmpeg` / `ffprobe`。

基础安装（歌词搜索、格式解析与核心数据结构）：

```sh
python -m pip install .
```

按需安装可选依赖组：

| 依赖组 | 用途 |
| --- | --- |
| `vocal` | Demucs 人声分离 |
| `bs-karaoke` | BS Karaoke 主唱分离 |
| `japanese` | Wav2Vec2 CTC / MMS_FA 词级对齐 |
| `whisper` | Whisper 转写与行级对齐 |
| `dev` | 单元测试与打包构建 |

例如同时安装人声分离与词级对齐依赖：

```sh
python -m pip install -c constraints.txt '.[vocal,bs-karaoke,japanese]'
```

`constraints.txt` 给出各模块的参考版本；若使用 NVIDIA GPU，请按本机 CUDA 环境安装对应版本的 PyTorch / TorchAudio。所有模型路径均指向本地目录，权重需由使用者自行按对应许可准备。

## 快速上手

### Python 接口

以词级对齐为例，先准备人声与人声片段，再依次执行声学推理、CTC 解码与结果整理：

```python
from lazulite.audio import BSKaraokeConfig, BSKaraokeSeparator, decode_audio
from lazulite.audio.preparation import prepare_vocals
from lazulite.alignment.japanese.backend import open_backend
from lazulite.alignment.japanese.run import compute_acoustics, decode_japanese, finalize_japanese

separator = BSKaraokeSeparator(BSKaraokeConfig("/models/bs-karaoke", device="cuda"))
try:
    prepared = prepare_vocals(decode_audio("song.m4a", sample_rate=44100, channels=2), separator=separator)
finally:
    separator.close()

backend = open_backend("/models/NextFire-karaoke-ja-Latn", device="cuda")
try:
    acoustic = compute_acoustics(prepared, backend=backend)
finally:
    backend.close()

# request 为调用方构造的 JapaneseAlignmentRequest
decoded = decode_japanese(acoustic, request)
result = finalize_japanese(decoded)
```

- 默认使用 Demucs 分离时，可将 `BSKaraokeSeparator` 替换为 `DemucsSeparator(DemucsConfig(device="cpu"))`；如果已经有分离好的人声文件，可直接调用 `prepare_existing_vocals(decode_native_audio("vocals.wav"))`。
- `open_backend` 会自动识别本地目录是 Hugging Face Wav2Vec2 CTC 模型（如 NextFire）还是 TorchAudio `MMS_FA` 模型。
- 行级对齐入口为 `AlignmentPipeline(transcriber=WhisperBackend(...)).run_prepared(prepared, document)`。

### 命令行

```sh
# 搜索在线歌词
lazulite lyrics search song.m4a --output candidates.json

# 词级对齐（从原曲分离或直接传入已分离的人声）
lazulite align-japanese --audio song.m4a --ttml lyrics.ttml --model-path /models/NextFire-karaoke-ja-Latn --output result.json
lazulite align-japanese --audio song.m4a --separator bs-karaoke --vocal-model-repo /models/bs-karaoke --ttml lyrics.ttml --model-path /models/MMS_FA --strip-tone-digits --output result.json
lazulite align-japanese --vocals vocals.wav --targets units.json --model-path /models/NextFire-karaoke-ja-Latn --output result.json

# 行级对齐
lazulite align lyrics.ttml --audio song.m4a --backend whisper --model-path /models/whisper --mode auto --output lines.json
```

命令行支持通过 `--acoustic-output` / `--acoustic-input`、`--decode-output` / `--decode-input`、`--transcriptions-output` / `--transcriptions-input` 显式保存并复用中间阶段结果，方便在修改读音或后处理参数后跳过重复推理。

## 更多文档与许可

- 各模块职责、阶段数据契约、软代价参数与取消/内存约定见 [架构与接口说明](docs/architecture.md)。
- 本地测试运行方式与自备素材评估见 [测试说明](tests/README.md)。
- 假名处理与部分可选软代价实现改编自 [MiliKara](https://github.com/BilyHurington/MiliKara)（MIT），完整第三方署名和许可声明见 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)。注意 [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) 模型采用 CC-BY-NC-SA-4.0 协议，禁止商用。
