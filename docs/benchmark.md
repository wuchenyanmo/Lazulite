# 整曲基准

`tools/benchmark.py` 以 `tests/fixtures` 中的三首整曲音频为输入，读取同名 TTML 作为评估真值。DP 输入会清除 TTML 的人工时间；它们不会被送入自动对齐器。每行按稳定 `line_id` 比较，端点缺失计入覆盖率分母。

运行示例：

```bash
python tools/benchmark.py \
  --backend qwen --model-path /models/Qwen3-ASR-0.6B \
  --revision <固定revision或权重SHA256> --mode dp \
  --output benchmark.json
```

输出逐曲记录音频 SHA256、运行时与 RTF、matched coverage、未解决行数、端点 MAE/P50/P90，以及 `≤0.25s` 与 `≤0.50s` 比例。Qwen 没有 token 时间，DP 结果使用声学窗口，因此该 benchmark 不能解释为逐字 forced-alignment 准确率。

首次重构基线将在完成同一固定模型、固定环境的三曲运行后记录在本页；在此之前不编造绝对阈值，也不以人工裁剪片段报告整曲准确率。
