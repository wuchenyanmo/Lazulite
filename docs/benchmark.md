# 整曲基准

`tools/benchmark.py` 以 `tests/fixtures` 中的三首整曲音频为输入，读取同名 TTML 作为评估真值。DP 输入会清除 TTML 的人工时间；它们不会被送入自动对齐器。每行按稳定 `line_id` 比较，端点缺失计入覆盖率分母。

运行示例：

```bash
python tools/benchmark.py \
  --backend qwen --model-path /models/Qwen3-ASR-0.6B \
  --revision <固定revision或权重SHA256> --mode dp \
  --output benchmark.json
```

输出是一个 JSON 对象，包含模型加载时间、模式与注入偏移，以及逐曲记录。每曲会保存音频 SHA256、总时长、热转写时长、RTF、CUDA allocated/reserved 峰值、后端及所有有效参数、逐行结果和诊断信息。`metrics.start` 与 `metrics.end` 分别保存 MAE、P50、P90、`≤0.25s`、`≤0.50s` 和完整覆盖率；端点缺失会留在覆盖率分母中。

`dp` 会清除全部 TTML 人工时间，防止真值泄漏。`offset`、`hybrid` 与 `auto` 则在每个真值端点减去 `--source-shift`（默认 4 秒），构造可验证偏移恢复能力的受控输入。逐行结果和原始 JSON 应与提交一同保存，避免只保留汇总数字。Qwen 没有 token 时间，DP 结果使用声学窗口，因此该 benchmark 不能解释为逐字 forced-alignment 准确率。

## 2026-09-06 重构基线

运行：Qwen/Qwen3-ASR-0.6B（`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`）、qwen-asr 0.0.6、Torch 2.14.0+cu130、Python 3.12.3、RTX 4090 49140 MiB、Demucs `htdemucs`、DP 模式。三首均从整曲解码、分离和声学分段开始；没有将 TTML 时间输入 DP。

| 素材 | 用时 | RTF | 覆盖率 | 未解决行 | 端点 MAE | P90 | ≤0.25s / ≤0.50s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POWDER SNOW (Live Ver.) | 46.119s | 0.1028 | 100.00% | 0 | 1.532s | 2.681s | 30.00% / 40.00% |
| 夜奏花 | 25.227s | 0.0715 | 96.97% | 1 | 5.694s | 19.115s | 39.06% / 45.31% |
| 鳥の詩 | 38.284s | 0.1037 | 100.00% | 0 | 1.407s | 4.140s | 33.33% / 57.14% |

这里的覆盖率是具有开始和结束候选的行数占比；端点指标只对具有可比较端点的结果统计。Qwen 路径没有 token 时间，因而这些是声学 core/插值的自动行级基线，尤其不应解读为逐字、mora 或 forced-alignment 精度。夜奏花的长尾误差也说明当前结果只能作为后续算法变更的比较基线，不能设置未经验证的绝对质量门槛。
