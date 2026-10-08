# Lazulite 0.2 模块与接口

Lazulite 只计算与返回候选，不持有编辑器项目，不合并旧端点，不保存或修改 TTML。
Python 3.12+，原轴时间为绝对秒；音频区间为整数样本半开区间，解码样本 0 对应原曲时间 0。

## 模块关系

- `models`：音频、准备产物、行级/日语输入与结果 dataclass。
- `audio/preparation.py`：分离＋单一声学分析实现，或显式外部完整人声/分析；`concat.py` 只负责 core 样本并集与双向坐标。
- `transcription`：实际请求窗口、后端证据、端点整理、共享风险公式。Whisper 与 Qwen 适配各自能力。
- `alignment/line.py`、`offset.py`：行级 DP 与 offset 接受门控。
- `alignment/japanese`：固定 profile、NextFire、全局 CTC、提示校准、局部建议。无 Whisper/转写依赖。
- `pipeline.py`：明确的行级组合；`cli.py`：输入校验、资源装配与生命周期。
- `cache.py`：调用方指定目录的分阶段 JSON/数组缓存；旧 schema 失效。

基础导入与 CLI help 不加载 Torch 或权重。NumPy 向量化 CTC 是本轮加速实现，不依赖研究运行时或 monkey patch。

## 准备与独立调用

```python
from lazulite.audio.io import decode_audio
from lazulite.audio.preparation import prepare_vocals, read_prepared, write_prepared
from lazulite.audio.separation import DemucsSeparator, DemucsConfig
from lazulite.alignment.japanese.backend import NextFireBackend, NextFireConfig
from lazulite.alignment.japanese.run import align_japanese, JapaneseConfig
from lazulite.models.japanese import JapaneseAlignmentRequest
from lazulite.cache import FileCache
from pathlib import Path
import json

separator = DemucsSeparator(DemucsConfig(device="cuda"))
try:
    prepared = prepare_vocals(
        decode_audio("song.m4a", sample_rate=44100, channels=2),
        separator=separator, cache=FileCache("cache"),
    )
finally:
    separator.close()
write_prepared(prepared, "prepared")
prepared = read_prepared("prepared/manifest.json")
request = JapaneseAlignmentRequest.from_dict(json.loads(Path("units.json").read_text()))
backend = NextFireBackend(NextFireConfig(Path("/models/NextFire"), "cuda"))
try:
    result = align_japanese(prepared, request, backend=backend,
                            config=JapaneseConfig(hint_mode="auto"), cache=FileCache("cache"))
finally:
    backend.close()
```

已有完整人声使用 `prepare_existing_vocals(stem, original_id=..., original_duration=...)`。
已有分析还须传入 `analysis` 和说明算法、配置及 reference 条件的 `analysis_identity`；
人工片段传入 `segments`，自动分析保留原样。无 reference 不等同于原混音参考分析。

行级独立调用：`AlignmentPipeline(transcriber=whisper, cache=cache).run_prepared(prepared, document, config=PipelineConfig(mode="dp"))`。
可分别导入 `prepare_existing_vocals`、`analyze_vocals`、`align_lines` 使用单阶段能力。

## 日语 JSON

```json
{
  "schema": "lazulite-japanese-request-v1",
  "lines": [{
    "line_id": "line-a",
    "text": "海へ",
    "start_hint": {"time": 0, "source": "edited"},
    "units": [
      {"unit_id": "sea", "source_range": [0, 1], "reading": "うみ", "reading_language": "ja", "reading_source": "user"},
      {"unit_id": "to", "source_range": [1, 2], "reading": "え", "reading_language": "ja", "reading_source": "user"}
    ]
  }]
}
```

`source_range` 使用原文 Unicode code point；分组由调用方提供，可有多个单元共享同一正文范围。
可空 `reading_range` 区分同一来源读音中的片段。未知读音用空串保留目标，不能自动猜汉字读音。
日语是 `ja`，拉丁是 `en` 或 `latin`；不按字符外观偷偷切换语言。
提示来源为 `imported/edited/automatic/unknown`，同路线自动提示应声明 `route="japanese"`。

结果 `lazulite-japanese-result-v1` 保留每个 ID 和输入快照，分开返回：

- `raw_timing`、`character_path`、`fragments` 与拼接轴区间；跨删段的外包络只能作诊断。
- `proposed_timing` 的两个独立端点（时间、来源、依据 ID）。`null` 只表示没有新建议。
- `complete/partial/unresolved` 表示覆盖，不表示正确率；覆盖同时统计 profile 派生和额外补全。
- 校准报告、首遍/第二遍身份、采用策略、结构化问题、模型分数定义。

正常双端保持原样。缺失插值只能发生在同一保留片段内；跨缝存在多个合理 fragments 时不无条件选最长者。
CTC 默认额外内存预算 512 MiB（回溯、滚动工作数组和输出路径），超限抛 `MemoryError`。

## 行级证据与状态

`RequestWindow` 记录独立请求 ID、父声学 ID、实际样本窗口和可空 core 交集。
Whisper 原始数组与生成步证据不变，后继派生/声学端点各自标来源；生成截断一直保留 issue。
概率摘要是正文生成 token 概率的算术均值、最低值与低于 0.35 的比例；`confidence` 的几何均值另保留，不是正确率。
`EvidenceConfig("whisper-lexical-v1")` 是显式对照策略，DP、offset、历史上下文共用一个函数；默认 `generic`。
Qwen 缺少该摘要时不适用 Whisper 概率门槛。

`LineAlignmentResult` 返回覆盖状态/统计/问题及 `offset_accepted`，无 `reliable` 字段。
每行提供起止来源、父片段 ID 和实际请求 ID。模型依然可能漏配或错配，完整覆盖不保证时间准确。

## 缓存、取消和发行边界

准备身份包含实际 PCM、分离权重及有效参数、声学分析版本/reference 条件。
日语声学缓存不含读音，解码缓存含完整请求与 profile；补全不重新推理。
Whisper 缓存记录精确 PCM、实际窗口、实际上下文和生成参数，改 DP 配置复用转写。
没有可信内容身份的外部后端不缓存模型产物。数组验证散列，完整写入后才发布 manifest。

取消回调是任务局部的；在准备阶段边界、声学块间、每 128 CTC 帧和转写请求间检查。
取消抛出 `events.Cancelled`，不会返回或发布最终成功结果，已完整生成的可复用中间缓存可保留。
调用方负责 `close()`；后端不会自动替换失败模型。

Linux CPU/CUDA 是本轮实际验证环境；Windows/macOS 的安装和模型路径、MPS 均须另行实测。
Studio 接入不属于本库变更，0.2 不提供旧接口或旧缓存迁移层。
