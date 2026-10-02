# 架构与接口

Lazulite 使用 `src/lazulite` 包布局，提供可独立调用的处理阶段。

| 模块 | 职责 |
| --- | --- |
| `models/` | 音频、歌词、转写和对齐的数据约定 |
| `audio/` | 元数据、PCM 解码、人声分离与声学分片 |
| `lyrics/` | LRC/TTML 投影、文本匹配、在线搜索与候选排序 |
| `transcription/` | 转写协议、分片调度、Whisper/Qwen 与 token 分布 |
| `alignment/` | 行级 DP、源时间偏移与独立 forced alignment |
| `pipeline.py` | 组合注入的分离器、转写器、缓存和事件回调 |

基础包提供轻量数据与 API；模型后端按需创建，通过 `close()` 释放实例资源。

## 数据约定

- 时间采用整曲绝对秒，未知值为 `None`；结果标注模型、声学窗口或插值来源。
- Whisper 的 Unicode 字符组从首个 BPE token 边界开始；Qwen 提供正文 token 概率，位置关联声学分片。
- `confidence=None` 表示缺失，`0` 表示有效零值；评分使用已有证据。
- `offset` 要求每行具备源开始时间；`hybrid` 用 DP 匹配约束偏移锚点。
- 缓存使用音频、模型和配置身份；完整分布产物缺失时重新计算。

## 声学与调度

分离入口应提供原始声道信息；CLI 的 `align` 入口解码为 44.1 kHz 双声道。内存重采样使用 FFmpeg SWR 带限滤波（filter_size=32、phase_shift=10、cutoff=0.97），保留样本零点，输出长度向上取整；ASR 边界才转换为 16 kHz 单声道。

`VocalConfig` 使用采样点参数 `frame_length=2048`、`hop_length=256`，其余时长以秒配置。活动曲线、谷底切分及 presence/off-center 评分都基于主唱信号；立体声主唱信号为 `mid - abs(side)`。单声道的侧置风险为 `None`。`analyze_vocals(..., reference=original_audio)` 显式关联原曲参考；未提供时采用满幅参考 1.0。

`VocalAnalysis.is_vocal` 和 `vocal_seconds` 表示整曲判定：RMS 2048/512、相对参考 -30 dB，活动至少 5 秒且占整曲至少 20%。局部分片存在不意味着整曲通过。Pipeline 未通过时保留输入歌词为 unresolved，不调用转写后端，不修改媒体文件。

`offset`/`auto` 先按 section 选择最多 3 个高质量窗口转写；auto 不可靠才补齐完整 DP。只有窗口、timing 和 context 请求相同才复用证据。偏移默认至少 3 个总锚点、每段至少 2 个；最终按 DP section 归属分配。已知但不可靠的 section 保留 unresolved；缺 section 的补齐标注 `inferred_section_offset`。多行有真实时间单位时搜索单调边界，没有单位时保留明确标注的粗插值。`AlignedLyricLine.matched_text` 保存行级匹配正文，末单位结束未知时行结束仍为 `None`。

`TranscriptionConfig` 从 `lazulite.transcription` 导入，支持 `none`、`previous`（默认）、`hint`、`hybrid`。Whisper 不请求细时间时可使用任务内滚动上下文；hint/hybrid 显式启用。细时间请求采用无 prompt 路径，直接同时请求 context 与细时间会拒绝。Qwen 当前不支持 context/hint。上下文不会跨任务保留。

结果缓存算法版本为 `pipeline-result-v2-preserved-algorithms`，旧版结果不会命中。`PipelineResult.execution` 保存有效参数、实际选片/请求窗口、timing 请求和输入处理方式；转写 metadata 保存实际 context。更改 prompt、声学配置或后端身份均改变缓存键。

## 歌词与搜索

明确角色前缀在文本/LRC 导入时提取，在 LRC 导出时还原；含糊的冒号正文保留。翻译独立解析，不应用正文的角色/元数据过滤，也不自动删括号；`clean_translation` 是显式清理操作。

搜索最多并发 4 个来源，收集后稳定排序；全部来源请求失败抛错，正常空结果仍为空。内容评分使用每个可用 ASR 片段的最佳歌词匹配及繁简视图，按 70/30 合成；无可比较 confidence 时使用非加权文本项。模型置信度与重复度、密度、声学质量启发式分开处理。候选存在总分 ≥0.55 的集合时仅返回该集合，依次优先真实时间、来源优先级、总分和源内排名。

QQ 使用 smartbox 搜索并补全歌曲详情，再请求 musicu 非加密逐行歌词和翻译，必要时回退旧逐行接口。该路径不把 QRC 密文交给 LRC 解析。

验证资料：[契约抽检](backend_runtime_smoke.md)、[冻结样例](alignment_regression.md)、[整曲基准](benchmark.md)。
