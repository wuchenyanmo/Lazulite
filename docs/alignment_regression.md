# 行级 DP 冻结样例

[alignment_regression.json](../tests/fixtures/alignment_regression.json) 保存通用转写、声学窗口和预期行结果。[test_alignment.py](../tests/test_alignment.py) 检查行 ID、时间、来源和分片关联。

| 行 | 输入证据 | 预期结果 |
| --- | --- | --- |
| L1 | s1 / 第一行 / confidence=None | 10–13s，`acoustic_core` |
| L2 | s2 / 第二行 / confidence=0 | 20–23s，`acoustic_core` |
| L3 | 待匹配歌词 | 时间为 null，`unresolved` |

该样例固定当前行为，覆盖缺失置信度、有效零值和保留待匹配行。

运行方式见[测试说明](../tests/README.md)，接口见[架构说明](architecture.md)。
