# 测试与素材

运行 CPU 测试：

```bash
python -m pytest
```

`fixtures/` 仅保存自造 DP 冻结样例。解析测试现场构造 TTML，音频测试用 FFmpeg 生成正弦波；不依赖真实歌曲或外部工作区素材。此前三首歌曲及其网络歌词已弃用并从当前工作树移除。模型推理由显式提供自备素材的 benchmark 命令执行。

整曲 DP 评估使用歌词正文，人工时间用于计算误差。offset/hybrid 评估使用注入已知偏移的源时间。

方法与结果见[整曲基准](../docs/benchmark.md)，数据约定见[架构说明](../docs/refactoring.md)。
