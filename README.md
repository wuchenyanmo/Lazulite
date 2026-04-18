## Lazulite: *Lyric Alignment* with isolated transcription and robust time estimation
这是一个用于“歌曲音频与已有歌词对齐”的 Python 项目。现在可用的开源asr模型都没有能从头转写歌词的能力，所以这个项目只实现了将音频与本地/在线获取的歌词对齐的功能，并且只能做到行级对齐，但是内部模块的功能可能可用于从头转写歌词的前处理。
目前对日语和粤语歌曲进行了少量测试，效果很好，即使在转写结果大部分全错的情况下依靠声学分析的分句也能可靠地利用少量正确转写结果对齐，其他语言暂未测试

## 功能概览

- 读取音频元数据，自动搜索歌词，或直接使用本地歌词
- 使用 Demucs 分离人声，并按人声活动做句级分片，根据双声道偏离中置的程度和人声能量评估片段分数
- 使用 Whisper 对高价值片段转写
- 对齐歌词，使用两种对齐模式：
  - `offset-only`：通过高置信度的anchor片段估计已有歌词与本地音频的整体或分段时间偏移
  - `dp`：使用 token 时间戳和单调动态规划做文本约束对齐
  - `auto`：默认值，先使用`offset-only`尝试估计偏移，anchor片段数过少或偏移不一致时回退至`dp`模式

## 安装
1. 参考PyTorch 安装说明：https://pytorch.org/get-started/locally/ ，
按自己CUDA版本和平台手动安装`torch`、`torchaudio`、`torchvision` 以及 `torchcodec`。
示例（CUDA 12.8）：
```bash
pip install torch torchaudio torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
```

2. 安装其余依赖：
```bash
pip install -r requirements.txt
```

3. 安装 FlashAttention-2 加速推理（可选）：
参考FlashAttention仓库：https://github.com/Dao-AILab/flash-attention

## 用法

使用本地lrc歌词：

```bash
python -m Lazulite example.m4a \
  --lyric-path example.lrc \
  --language ja \
  --output-lrc example.lrc \
```

在线获取歌词：

```bash
python -m Lazulite example.m4a
```

使用 `python -m Lazulite --help` 查看更多参数和用法，也可作为 Python 库导入并调用

## 性能
默认模型是 Whisper-large-v3 ，参数量约1.5B，首次运行时会自动从 Hugging Face 下载模型
在RTX 4090上测试，`offset-only`模式单首歌曲耗时30秒，峰值占用显存约5G，若回退到`dp`模式则增加至1分钟，峰值占用显存约8G
