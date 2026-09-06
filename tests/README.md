# 重构测试素材

`fixtures/` 包含三首用户提供的歌曲、清洗后的人工 TTML 和秒制 manifest。旧 `test/` 的生成输出已删除。

执行文档：`/home/wangzeyu/wangzeyu-nvme1/lyric-workflow/docs/lazulite_refactoring_guide.md`。模型实测及清洗记录位于该目录的 `evidence/`。

这是整曲本地 benchmark 素材，不自动加入普通单元测试，不打入 wheel。时间真值不得泄漏进待评估的 DP 输入。
