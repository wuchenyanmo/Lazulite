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

验证资料：[契约抽检](backend_runtime_smoke.md)、[冻结样例](alignment_regression.md)、[整曲基准](benchmark.md)。
