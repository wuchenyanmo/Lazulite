# Lazulite 架构与接口说明

本文档说明 Lazulite 的模块划分、各计算阶段的数据约定以及资源控制边界。所有对外时间统一使用以原曲起点为 0 的秒数，音频样本区间采用半开区间 `[start, end)`。

## 模块划分

| 模块 | 职责 |
| --- | --- |
| `models/` | 音频、人声分析、转写结果、行级与词级对齐请求/结果数据结构 |
| `lyrics/` | 网易云/QQ/酷狗/LRCLIB 在线歌词搜索、RapidFuzz 匹配打分、文本规范化、LRC 与 TTML 解析 |
| `audio/` | FFmpeg 音频解码与重采样（`io.py`）、Demucs 分离（`separation.py`）、BS Karaoke 主唱分离（`bs_karaoke.py`）、声学活动分析（`activity.py`）、人声片段准备（`preparation.py`）及片段拼接映射（`concat.py`） |
| `alignment/japanese/` | CTC 词级对齐管线：多语言输入准备（`input.py`、`kana.py`、`profile.py`）、本地模型加载与分块推理（`backend.py`）、Viterbi CTC 解码（`ctc.py`）、行首提示校准（`calibration.py`）、端点整理（`completion.py`）与分阶段入口（`stages.py`、`run.py`） |
| `transcription/` | Whisper 转写窗口切分、后端调用、词级时间戳整理与质量统计 |
| `alignment/line.py`, `offset.py` | 通用跨语言行级动态规划匹配与时间偏移估计 |
| `pipeline.py`, `exchange.py`, `cli.py` | 行级组合管线、阶段结果文件读写与命令行入口 |

基础导入、歌词搜索、格式解析与阶段文件读写均不加载 PyTorch，重型模型依赖只在对应阶段按需导入。

## 人声分离与片段准备

`DemucsSeparator` 和 `BSKaraokeSeparator` 实现统一的 `separate(AudioBuffer) -> AudioBuffer` 接口：

- `DemucsSeparator`：支持默认权重或指定本地离线模型目录，适合大部分歌曲。
- `BSKaraokeSeparator`：加载本地 `bs_roformer_karaoke_frazer_becruily.ckpt` 与对应 YAML 配置，按固定窗口加权重叠拼接输出 44.1 kHz 双声道主唱，对和声和背景噪声分离更干净。支持 CPU 与 CUDA。

入口函数：
- `prepare_vocals(original, separator=...)`：先分离人声，再以原曲为电平参考做声学活动分析，划分人声片段。
- `prepare_existing_vocals(vocals, analysis=..., segments=...)`：直接消费已分离的人声。若外部已提供 `analysis` 或人工调整后的 `segments`，则直接采用，不会重新覆盖。

`audio/activity.py` 提供 `build_activity_profile(audio, config=...)` 与 `analyze_vocals(audio, config=..., profile=...)`。调用方如果既需要人声能量曲线（`main_rms`）又需要人声片段分析，可先计算一次 `profile` 再传入 `analyze_vocals` 复用。

## CTC 词级对齐阶段

词级对齐拆为三个独立阶段，也可通过 `align_japanese(...)` 一次串联执行：

1. **声学推理（`compute_acoustics`）**：
   - 将选中的人声片段按采样率拼接，记录拼接轴与原曲轴之间的样本区间映射（`ConcatMap`）。
   - 调用声学后端计算归一化对数概率矩阵（`[frames, vocab_size]`）与帧映射（`FrameMap`），按需一并计算逐帧休止强度 `rest`。
   - 返回 `AcousticResult`，可供后续不同读音或参数反复解码。
2. **输入准备与 CTC 解码（`decode_japanese`）**：
   - `input.prepare_input` 接收 `JapaneseAlignmentRequest`（`lazulite-japanese-request-v1`）、模型词表与 `InputOptions`：
     - 过滤非发音标点，保留发音长音符与待编码字符，维护每个内部片段回溯原单元的映射。
     - `latin_normalize`（默认开启）：日语假名按行内上下文转为 Hepburn 罗马字，其余文字由 `uroman` 转为拉丁字母；关闭时保留原字形直接查词表。
     - `strip_tone_digits`（默认关闭）：移除每个词或空白分隔片段末尾的连续声调数字。
   - 执行首遍 CTC Viterbi 解码；当启用行首提示校准（`hint_mode="auto"`）且存在外部参考时间时，再结合锚点提示执行第二遍解码。
   - 返回 `DecodeResult`，保留输入请求、词表编码范围、首遍与采用路径以及坐标映射。
3. **投影与端点整理（`finalize_japanese`）**：
   - 将拼接轴帧区间回映到原曲时间轴，拆分跨人声片段接缝的区间。
   - 当 `completion=True`（默认）时，根据同行相邻边界与片段时长占比补全缺失或被切碎的单元起止，同时在结果中完整保留原始路径 `raw_timing` 与整理后的 `proposed_timing`。

### 本地对齐模型后端

- `alignment.japanese.backend.describe_model(path)` 检查本地模型目录的小型配置与权重文件存在性，返回模型类型、词表、blank ID、帧步长和所需文件列表。
- `open_backend(path, device=...)` 自动选择对应后端：
  - `NextFireBackend`：Hugging Face Wav2Vec2 CTC 目录（含 `config.json`、`preprocessor_config.json`、`vocab.json` 与权重文件）。
  - `MMSFABackend`：TorchAudio `MMS_FA` 目录，通过 `alignment-model.json` 声明 `{"type": "torchaudio-mms-fa", "weights": "model.pt"}`。
- 两者共用 20 秒核心块加前后 2 秒上下文的分块推理流程。`MMS_FA` 作为通用模型通常会出现词尾时间偏早，建议在上游后处理中配合静音检测延长结束时间。

### 可选软代价与接缝代价

CTC 在人声片段拼接接缝后的首个声学帧，默认对同行状态转移施加一次接缝软代价，避免跨空白长距离误连。

`JapaneseConfig` 另外提供三项源自 MiliKara 的可选软代价（均默认关闭，启用时从路径得分中按帧扣除，设单帧时长为 `Δt` 秒、相对休止强度为 `rest ∈ [0, 1]`）：

| 开关 | 作用状态 | 每帧扣分 |
| --- | --- | --- |
| `line_blank` | 同行相邻目标之间的 CTC blank | `Δt` |
| `rest_blank` | 同行相邻目标之间的 CTC blank | `4 × rest × Δt` |
| `rest_token` | 所有目标 token 状态 | `25 × rest × Δt` |

消融测试显示这三项在多数样本上提升有限，部分条件下甚至会劣化边界，使用前建议结合素材评估。

## 行级对齐与转写阶段

行级路线由 `AlignmentPipeline` 组织，也可拆分调用 `build_transcription_inputs`、`run_transcription_stage`、`align_lines` 与 `align_with_offset`：

- `mode="dp"`：不依赖原歌词时间，使用 Whisper 转写人声片段并通过动态规划匹配各行。
- `mode="offset"`：在歌词已有参考时间时，通过预选窗口估算整体或分段偏移。
- `mode="auto"` / `"hybrid"`：优先验证偏移估计，条件不足时回退到完整窗口动态规划。

转写结果仅作为定位参考，最终结果保留原歌词正文并记录每行的时间来源与置信信息。

## 阶段文件交换、取消与内存控制

- `exchange.py` 负责读写显式指定的声学 `.npz`、解码 `.json` 与转写阶段 `.json` 文件，并在读取时校验数据结构。库本身不管理磁盘缓存目录。
- 长时间操作在阶段切换、分块推理、转写窗口和 CTC 帧边界检查取消回调，触发后抛出 `events.Cancelled`。
- CTC 解码默认将额外内存上限控制在 512 MiB（可通过 `max_decode_memory_bytes` 调整），超限时抛出 `MemoryError`。调用方负责在模型使用完毕后调用 `close()` 释放显存与内存。
