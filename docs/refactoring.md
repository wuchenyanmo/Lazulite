# 重构执行入口

完整指导位于 [lazulite_refactoring_guide.md](/home/wangzeyu/wangzeyu-nvme1/lyric-workflow/docs/lazulite_refactoring_guide.md)，模型实测见 [asr_model_probe_report.md](/home/wangzeyu/wangzeyu-nvme1/lyric-workflow/docs/asr_model_probe_report.md)。

清洗后的三首测试素材已经放入本项目 `tests/fixtures`。请先阅读指导及真实输出，再开始重构；不保留旧字段/API兼容层。

2026-09-06 当前决定：token 概率分布是必需能力，使用官方 Python 推理路径：Whisper 的 Transformers generate 与 Qwen 官方模型的 generate。两个后端均保留逐 token 分数、top-k 和显式全分布导出能力。依赖、接口、实现步骤及必要测试以完整重构指导第 5 节为准。

[统一引擎评估](/home/wangzeyu/wangzeyu-nvme1/lyric-workflow/docs/unified_asr_runtime_evaluation.md) 仅保留调研记录；transcribe.cpp/GGUF 方案已撤回，不纳入本轮重构。
