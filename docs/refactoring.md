# 重构说明

Lazulite 现在是采用 `src` 布局的小写 Python 包 `lazulite`。旧的大写包、旧 requirements 文件和 FlashAttention 专用依赖已删除；项目不提供兼容 import 或旧命令别名。

## 模块边界

- `models/`：音频、歌词、转写、对齐的模型无关数据约定。
- `audio/`：元数据、显式 PCM 解码、人声分离实例和声学活动分片。
- `lyrics/`：LRC/TTML 投影、文本规范化、在线来源、候选获取和内容重排。
- `transcription/`：`Transcriber` 协议、分片调度、token 分布以及按需加载的 Whisper/Qwen 后端。
- `alignment/`：行级单调 DP、源时间偏移与独立 forced alignment 协议。
- `pipeline.py`：只装配调用方注入的 separator、后端、缓存和事件回调；不搜索网络、不写回媒体。

基础安装导入 `lazulite` 不会加载 Torch、Transformers、Demucs 或模型权重。具体后端由调用方按需创建并通过 `close()` 结束生命周期。

## 关键约定

- 时间一律为相对整曲起点的秒数；未知时间为 `None`，不以窗口或均分时间伪造模型证据。
- Qwen 生成 token 概率但没有 token 时间；Whisper 使用生成步前缀修正后的时间边界。跨 token Unicode 字符以首 token 边界开始。
- `confidence=None` 与合法的 `0` 不同；缺失置信度和缺少细时间不会让候选自动失去资格。
- 显式 `offset` 要求每行都有源开始时间；`hybrid` 先运行 DP，再以其匹配结果限制偏移锚点。
- 文件缓存只在音频内容身份和资源身份可验证时启用；显式导出的完整分布文件丢失时，缓存会失效并重算。

## 验证资料

CPU 契约测试覆盖 token 分布数学、Whisper 前缀与 Unicode 边界、Qwen 自动语言头、forced 输入范围、DP/offset/hybrid 策略和缓存产物。整曲基准的输入原则、环境和结果记录在 [benchmark.md](benchmark.md)。

本项目不再链接或依赖任何特定机器上的外部说明、模型目录或测试工作流路径。
