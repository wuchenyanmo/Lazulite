# 重构测试素材

`fixtures/` 包含三首用户提供的歌曲、清洗后的人工 TTML 和秒制 manifest。旧 `test/` 的生成输出已删除。

这是整曲本地 benchmark 素材，不自动加入普通单元测试，不打入 wheel。时间真值不得泄漏进待评估的 DP 输入。

素材的使用边界、评估方法和当前结果见项目内 [docs/benchmark.md](../docs/benchmark.md)；核心架构约定见 [docs/refactoring.md](../docs/refactoring.md)。
