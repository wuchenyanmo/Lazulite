# 测试与素材

运行 CPU 测试：

```bash
python -m pytest
```

`fixtures/` 保存三首歌曲、人工 TTML、含 SHA256 的秒制 `manifest.json`，以及 DP 冻结样例。普通测试使用素材检查解析、解码和数据约定；模型推理由显式 benchmark 命令执行。

整曲 DP 评估使用歌词正文，人工时间用于计算误差。offset/hybrid 评估使用注入已知偏移的源时间。

方法与结果见[整曲基准](../docs/benchmark.md)，数据约定见[架构说明](../docs/refactoring.md)。
