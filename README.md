## Lazulite: *Lyric Alignment* with isolated transcription and robust time estimation
这是一个用于“歌曲音频与已有歌词对齐”的 Python 项目。现在可用的开源asr模型都没有能从头转写歌词的能力，所以这个项目只实现了将音频与本地/在线获取的歌词对齐的功能，并且只能做到行级对齐，但是内部模块的功能可能可用于从头转写歌词的前处理。
目前对日语和粤语歌曲进行了少量测试，效果很好，即使在转写结果大部分全错的情况下依靠声学分析的分句也能可靠地利用少量正确转写结果对齐，其他语言暂未测试

## 功能概览

- 读取音频元数据，自动搜索歌词（支持酷狗、QQ、网易、LRCLIB），或直接使用本地歌词
- 使用 Demucs 分离人声，并按人声活动做句级分片，根据双声道偏离中置的程度和人声能量评估片段分数
- 使用 Whisper 对高价值片段转写
- 对齐歌词，使用三种对齐模式：
  - `offset-only`：通过高置信度的anchor片段估计已有歌词与本地音频的全局或分段时间偏移
  - `dp-only`：使用 token 时间戳和单调动态规划做细粒度的文本约束对齐
  - `hybrid`：若有原始时间戳，根据`dp-only`的对齐结果重新估计全局或分段的时间偏移；若源歌词无时间戳，退化为`dp-only`
  - `auto`：默认值，优先使用`offset-only`尝试估计偏移，anchor片段数过少或偏移不一致时回退至`hybrid`模式

## 安装
1. 安装ffmpeg

2. 参考PyTorch 安装说明：https://pytorch.org/get-started/locally/ ，
按自己CUDA版本和平台手动安装`torch`、`torchaudio`、`torchvision` 以及 `torchcodec`。
示例（CUDA 12.8）：
```bash
pip install torch torchaudio torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
```

3. 安装其余依赖：
```bash
pip install -r requirements.txt
```

4. 安装 FlashAttention-2 加速推理（可选，仅支持Ampere以上架构GPU，即RTX 30系及以上）：
参考FlashAttention仓库：https://github.com/Dao-AILab/flash-attention

## 用法

使用本地歌词（lrc或txt格式，需要保证歌词顺序和完整性，可以没有时间戳）：
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

批处理模式（批量处理目录下的音频文件，不递归遍历）：
```bash
python -m Lazulite <dir>
```

对齐模式的选择：
- `offset-only`：如果你的显存是6G及以下，基本只能选择这个模式，除非你能接受溢出显存后极慢的转写速度。该模式对声学分句和转写的准确度都有较高的依赖，适合咬字清晰、断句清楚的歌曲，而且也依赖源歌词时间轴
- `dp-only`：由于需要返回token级的时间戳，显存占用会达到8G左右，该模式强依赖于声学分句的准确性，反而对转写质量没有太高要求，对一些断句位置很奇怪的日语歌效果较差，很适合中文歌曲
- `hybrid`：同样依赖源歌词时间轴，但是对声学分句准确度和转写质量的依赖大大降低，显存占用和`dp-only`模式相同
- `auto`：一般情况下可以达到不错的效果，但会默认歌词源的时间轴是可信的（即仅有整体或段落级的偏移），如果是极小众的歌曲，建议使用`dp-only`模式

使用 `python -m Lazulite --help` 查看更多参数和用法，也可作为 Python 库导入并调用

## 性能
默认模型是 Whisper-large-v3 ，参数量约1.5B，首次运行时会自动从 Hugging Face 下载模型

在RTX 4090上测试，`offset-only`模式单首歌曲耗时30秒，峰值占用显存约5G，若回退到`hybrid`或`dp-only`模式则增加至1分钟，峰值占用显存约8G
