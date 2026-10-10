# Lazulite 模块与接口

Lazulite 计算并返回候选。Python 3.12+；原轴时间为绝对秒，音频样本区间为半开区间，样本 0 对应原曲时间 0。

## 模块

- `models`：音频、完整人声、转写证据、行级和日语请求／结果。
- `audio/preparation.py`：原曲分离与完整人声分析；`concat.py`：core 并集和双向样本映射。
- `transcription`：请求窗口、后端生成、概率、上下文和细时间整理。
- `alignment/line.py`、`offset.py`：行级 DP 和 offset 门控。
- `alignment/japanese`：目标 profile、声学适配器、CTC、提示校准和端点整理。
- `pipeline.py`：行级组合；`exchange.py`：调用方指定文件的阶段序列化；`cli.py`：命令行资源装配。

基础导入、阶段消费与 CLI help 不加载 Torch。日语路线与转写／行级 DP 独立。

## 完整人声

`prepare_vocals(original, separator=...)` 分离后使用原曲参考分析；`prepare_existing_vocals(vocals, analysis=..., segments=...)` 消费完整人声和可选分析。时长默认来自音频，原轴零点固定为 0，业务来源 ID 可选。

`PreparedVocals` 保存波形、采样率、完整原轴时长、分析和选定片段。显式分析检查音频关联和范围。显式 `segments` 为人工选择；`None` 表示采用自动分析，空元组表示人工选择为空。自动明确无人声、未知自动判定和人工片段分别处理。波形不复制，调用方负责输入在计算期间保持不变。

## 日语阶段

`compute_acoustics(prepared, backend=..., include_activity=False)` 返回 `AcousticResult`：

- `[frames, columns]` 归一化自然对数概率；允许负无穷表示零概率，拒绝 NaN、正无穷和错误归一化。
- token 到列的映射、blank ID 和特殊 token。
- `FrameMap(sample_rate, stride_samples, receptive_samples, offset_samples)`；边界 `i` 的秒数是 `(offset + i × stride) / rate`。帧区间 `[a,b)` 转换为两个边界。
- `ConcatMap(sample_rate, sample_count, pieces)` 和原轴总时长；原轴、拼接轴区间均使用该采样率下的整数样本。
- 可选逐帧 rest 和实际运行描述。

`decode_japanese(acoustic, request, config=..., prepared=None, encoder=None)` 执行目标编码、首遍 CTC 与可选提示辅助第二遍。休止代价需声学结果中的活动；缺少时可提供完整人声计算。传入波形时必须匹配声学坐标。只消费概率与活动不需要模型或原曲。

`DecodeResult` 保留目标请求、token 顺序、单元范围、首遍／采用帧路径、分数、坐标和校准记录。`finalize_japanese(decoded, completion=True)` 单独执行原轴投影、长音与端点整理。`align_japanese(prepared, request, backend=..., acoustic=..., decoded=...)` 为组合入口。已提供结果不重算相应阶段；冲突来源、请求变化和非法结构直接失败。

`NextFireBackend` 从本地配置推导 Wav2Vec2 CTC 元信息，使用 FP32、整段配置归一化、20 秒核心块与前后 2 秒上下文，核心边界对齐实际帧网格且每帧仅覆盖一次。配置含输出 adapter 的架构需另行适配。自定义后端提供 `frames`、`vocab`、`blank`、`vocabulary_size`、`special_tokens`、`description`、`emissions(audio, cancel=..., event_sink=...)`。

日语请求 `lazulite-japanese-request-v1` 示例：

```json
{"schema":"lazulite-japanese-request-v1","lines":[{"line_id":"l","text":"海へ","units":[{"unit_id":"sea","source_range":[0,1],"reading":"うみ","reading_language":"ja","reading_source":"user"},{"unit_id":"to","source_range":[1,2],"reading":"え","reading_language":"ja","reading_source":"user"}]}]}
```

正文范围使用 Unicode code point；可空 `reading_range` 区分同一读音来源的片段。未知读音为空串，所有目标保留 ID。提示来源为 `imported/edited/automatic/unknown`，同路线自动提示声明 `route="japanese"`。

结果 `lazulite-japanese-result-v1` 分开保留 `raw_timing`、`proposed_timing`、`character_path`、原轴 `fragments`、拼接区间和问题。跨删段外包络仅作诊断；缺失插值只能在同一保留片段内。状态和覆盖统计不等于准确率。

## 转写与行级阶段

`build_transcription_inputs` 生成实际样本窗口和父片段关联，独立请求 ID 由片段 ID 与窗口样本边界组成。`run_transcription_stage` 返回 `TranscriptionStage`，包含片段、语言、细时间开关、上下文策略、后端能力和完整结果。实际提示与生成条件写入每条证据；previous 历史只存在于本次调用。

`AlignmentPipeline.run` 可从原曲（注入 separator）或完整人声开始；`run_prepared` 使用明确分析和选择。`transcriptions` 字典的 `preselected` 与 `full` 分别表示 offset 预选和完整窗口条件。`auto` 接受 offset 时只需预选；回退时需完整条件。已提供阶段严格校验，缺失阶段仅在注入转写后端时计算。同条件无上下文的预选结果可在本次组合内与剩余窗口合并。

`align_lines`、`align_with_offset` 直接消费 `TranscriptionResult` 与分析。每行提供独立起止来源、父片段和请求 ID。Whisper 保存原始生成时间、Unicode/BPE 步映射、概率和停止原因，派生端点不改写原始证据。

## 文件、取消与资源

`exchange` 只读写指定的声学 NPZ、解码 JSON 和转写阶段 JSON；读取验证当前结构，不访问模型。大矩阵只由声学阶段或显式文件输出取得，普通最终结果不携带矩阵。

计算层不选择持久化位置，不自动发现或保存历史结果。调用方决定某份结果是否适用于本次输入和模型。后端 `close()` 释放实例资源；取消抛出 `Cancelled`，不返回最终成功结果。CTC 额外内存预算默认 512 MiB，实际所需空间超限即失败。
