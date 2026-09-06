# 整曲基准

`tools/benchmark.py` 使用三首整曲音频和同名人工 TTML，按稳定 `line_id` 评估行级时间候选。

运行示例：

```bash
python tools/benchmark.py \
  --backend qwen --model-path /models/Qwen3-ASR-0.6B \
  --revision REVISION --mode dp \
  --output benchmark.json
```

将模型路径和 `REVISION` 替换为实际目录及固定 revision 或权重 SHA256。JSON 保存模型加载时间、逐曲处理与热转写时长、RTF、CUDA 显存峰值、配置、逐行结果和诊断。

- `dp` 输入歌词正文，人工时间用于评估。
- `offset`、`hybrid`、`auto` 将源端点提前 `--source-shift` 秒（默认 4 秒），评估偏移恢复。
- `metrics.start/end` 分别保存 MAE、P50/P90 和 0.25s、0.50s 内的比例，以可比较端点统计；`complete_coverage` 以全部歌词行为分母。
- Qwen DP 采用声学 core 或插值时间。本页指标衡量行级候选，供人工复核和后续算法比较。

## 2026-09-06 重构验收基线

环境为 Python 3.12.3、qwen-asr 0.0.6、Transformers 4.57.6、Torch 2.14.0+cu130、RTX 4090（49140 MiB）与 Demucs `htdemucs`。分离器实际参数是 `device=cuda`、`split=true`、`overlap=0.25`。三首均从整曲解码、分离和声学分段开始。

模型权重身份已在本机核验：Qwen 0.6B 为提交 `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，`model.safetensors` SHA-256 为 `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`；Qwen 1.7B 两个分片的 SHA-256 分别为 `a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6` 和 `6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc`；Whisper large-v3 `model.safetensors` SHA-256 为 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`。

### DP：正文输入

三组原始逐行记录： [Qwen 0.6B](benchmarks/qwen-0.6b-dp-2026-09-06.json)、[Qwen 1.7B](benchmarks/qwen-1.7b-dp-2026-09-06.json)、[Whisper large-v3](benchmarks/whisper-large-v3-dp-2026-09-06.json)。

| 后端 / 素材 | 总用时 | RTF | 覆盖率 / 未解决行 | start MAE | end MAE | 热转写 | allocated / reserved 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 0.6B / POWDER SNOW | 35.369s | 0.0788 | 100.00% / 0 | 0.560s | 2.470s | 25.526s | 3178 / 3862 MiB |
| Qwen 0.6B / 夜奏花 | 25.795s | 0.0731 | 93.94% / 2 | 5.261s | 4.415s | 17.336s | 2855 / 3862 MiB |
| Qwen 0.6B / 鳥の詩 | 38.402s | 0.1041 | 100.00% / 0 | 0.405s | 2.005s | 29.390s | 2909 / 3862 MiB |
| Qwen 1.7B / POWDER SNOW | 52.942s | 0.1180 | 100.00% / 0 | 3.315s | 5.123s | 28.790s | 5573 / 6282 MiB |
| Qwen 1.7B / 夜奏花 | 26.542s | 0.0752 | 100.00% / 0 | 15.397s | 13.700s | 21.959s | 5251 / 6282 MiB |
| Qwen 1.7B / 鳥の詩 | 51.162s | 0.1386 | 97.62% / 1 | 5.240s | 7.415s | 35.623s | 5305 / 6286 MiB |
| Whisper large-v3 / POWDER SNOW | 220.740s | 0.4919 | 100.00% / 0 | 0.669s | 3.215s | 210.187s | 6677 / 7314 MiB |
| Whisper large-v3 / 夜奏花 | 127.522s | 0.3612 | 93.94% / 2 | 4.919s | 5.089s | 114.104s | 6374 / 7314 MiB |
| Whisper large-v3 / 鳥の詩 | 212.926s | 0.5769 | 97.62% / 1 | 1.141s | 2.920s | 204.952s | 6419 / 7314 MiB |

模型加载：Qwen 0.6B 9.404s、Qwen 1.7B 10.395s、Whisper large-v3 7.395s。完整覆盖率表示同时具有开始和结束候选的行数占比。

### offset / hybrid：已知 4 秒偏移

受控输入把每个 TTML 源端点统一提前 4 秒；测量指标仍相对原始 TTML 真值。原始逐行记录为 [offset](benchmarks/qwen-0.6b-offset-shift-4s-2026-09-06.json) 与 [hybrid](benchmarks/qwen-0.6b-hybrid-shift-4s-2026-09-06.json)，均使用 Qwen 0.6B。

| 模式 / 素材 | 覆盖率 / 未解决行 | start MAE | end MAE | 行时间来源 |
| --- | ---: | ---: | ---: | --- |
| offset / POWDER SNOW | 100.00% / 0 | 0.139s | 0.139s | `source_offset` |
| offset / 夜奏花 | 100.00% / 0 | 0.142s | 0.142s | `source_offset` |
| offset / 鳥の詩 | 100.00% / 0 | 0.102s | 0.102s | `source_offset` |
| hybrid / POWDER SNOW | 100.00% / 0 | 0.134s | 0.134s | `source_offset` |
| hybrid / 夜奏花 | 100.00% / 0 | 0.146s | 0.146s | `source_offset` |
| hybrid / 鳥の詩 | 100.00% / 0 | 0.108s | 0.108s | `source_offset` |

两种模式均恢复受控偏移，完整覆盖全部歌词行。hybrid 使用 DP 匹配约束锚点；这一结果衡量已知源时间的偏移恢复能力。
