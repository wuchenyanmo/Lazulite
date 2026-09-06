# 行级 DP 冻结回归

`tests/fixtures/alignment_regression.json` 是模型无关的迁移回归输入：两条无模型时间、其中一条置信度为合法 `0` 的转写证据，以及一条未匹配歌词。它冻结了声学 core 时间、分片关联和未解决行的逐行结果。

| 行 | 冻结输入证据 | 预期结果 | 差异判定 |
| --- | --- | --- | --- |
| L1 | s1 / `第一行` / confidence 缺失 | 10.0–13.0，`acoustic_core` | 完全一致 |
| L2 | s2 / `第二行` / confidence 为 0 | 20.0–23.0，`acoustic_core` | 完全一致；0 不得当作缺失 |
| L3 | 无文本匹配 | `null`，`unresolved` | 完全一致；不得静默丢行 |

`tests/test_alignment.py` 从该文件重建通用数据模型并逐字段比较。该快照用于隔离算法迁移回归；真实模型字段契约由 `test_backend_contracts.py` 覆盖，整曲效果由 [benchmark.md](benchmark.md) 记录，三者不互相替代。
