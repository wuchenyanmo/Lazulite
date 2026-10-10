# 测试与素材

安装开发依赖并运行 CPU 测试：

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

测试不需要模型权重、真实歌曲、Git 历史或外部工作区。解析测试现场构造 TTML，音频测试使用 FFmpeg 生成正弦波。
`fixtures/alignment_regression.json` 是自造的行级 DP 冻结样例；`fixtures/ctc_v1.py` 是用于验证关闭附加代价时路径不变的冻结 CTC 参考实现，只被测试加载，不进入运行时包。

## 自备素材评估

在 `--fixture-dir` 指定的目录中放置同名 `.m4a` 和 `.ttml` 文件；模型与素材路径均由调用者提供。在源码目录执行：

```bash
python -m pip install -c constraints.txt '.[whisper,vocal]'
python tools/benchmark.py --fixture-dir /path/to/fixtures \
  --backend whisper --model-path /models/whisper-large-v3 \
  --mode dp --output benchmark.json
```

DP 只使用歌词正文，人工时间仅用于计算误差；offset/hybrid 使用注入已知偏移的源时间（`--source-shift` 默认 4 秒）。输出包含实际配置、逐行结果、起止误差和覆盖率；`complete_coverage` 仅表示起止均有值，不保证边界准确。

接口见[架构说明](../docs/architecture.md)，行级样例见[冻结样例说明](../docs/alignment_regression.md)。
