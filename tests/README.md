# 测试与素材评估

## 运行单元测试

安装开发依赖后可直接运行全部 CPU 单元测试：

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

单元测试自带轻量合成音频与 TTML 测试数据，本地运行前只需确保系统已安装 `ffmpeg`。

## 自备素材评估

若希望在自己的歌曲数据集上评估行级对齐误差，可在 `--fixture-dir` 目录下放置同名的 `.m4a` 音频与 `.ttml` 歌词文件，然后运行评估脚本：

```bash
python -m pip install -c constraints.txt '.[whisper,vocal]'
python tools/benchmark.py --fixture-dir /path/to/fixtures \
  --backend whisper --model-path /models/whisper-large-v3 \
  --mode dp --output benchmark.json
```

- `dp` 模式仅使用歌词正文进行对齐，以参考时间计算误差。
- `offset` / `hybrid` 模式会将参考时间按 `--source-shift`（默认 4 秒）整体平移后测试纠偏效果。
- 输出 JSON 记录实际参数、逐行误差和覆盖率（`complete_coverage` 仅表示起止时间均已填入）。

更多模块与接口细节见 [架构与接口说明](../docs/architecture.md)。
