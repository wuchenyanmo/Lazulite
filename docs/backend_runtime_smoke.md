# 后端生成抽检

素材：`夜奏花` 的 1.0–9.0 秒片段。环境：qwen-asr 0.0.6、Transformers 4.57.6、Torch 2.14.0+cu130、RTX 4090。模型身份见[整曲基准](benchmark.md)。

抽检用于检查生成 ID、正文范围和时间边界。

| 后端 / 请求 | 输出 | 记录 |
| --- | --- | --- |
| Qwen 0.6B / 自动语言 | `夜空を見上げて。` | Japanese；10 个生成步，正文步为 3–8；正文置信度 0.873253 |
| Whisper large-v3 / ja | `夜空を見上げて` | 7 个时间单元；夜 1.000–2.020s，空 2.020–2.920s，を 2.920–5.560s |

Qwen 按正文生成步计分；Whisper 从字符组的首 token 边界取时。对应 CPU 用例见 [test_backend_contracts.py](../tests/test_backend_contracts.py)。

本页保存人工汇总；完整生成序列、原始时间数组与抽检命令仍待归档。端到端效果见[整曲记录](benchmark.md)。
