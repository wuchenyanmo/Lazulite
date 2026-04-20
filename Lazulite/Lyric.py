import re
from collections import Counter
from Lazulite.TextNormalize import RE_SPACES, normalize_text

# 基础 LRC 解析规则
RE_METADATA = re.compile(r'^\[([a-zA-Z]+):(.*)\]$')
RE_LYRICS = re.compile(r'^\[(\d{2,}:\d{2}\.\d{2,3})\](.*)$')
# 匹配日文歌词中常见的汉字(假名)注音，用于规范化时去掉括号内读音
RE_FURIGANA = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶ]+)\(([\u3040-\u30ffー・ ]+)\)')
RE_METADATA_SEP = re.compile(r'\s*[:：|｜]\s*')
RE_LYRIC_QUOTES = re.compile(r'[「」『』【】〈〉《》〔〕〘〙〚〛“”‘’"]+')

# 行首歌手/角色标记的几种常见形式
RE_SINGER_COLON = re.compile(r'^(?P<label>[^\s:：|｜]{1,24})\s*[:：|｜]\s*(?P<body>.+)$')
RE_SINGER_BRACKET = re.compile(
    r'^[\(\[（【「『<＜](?P<label>.{1,24}?)[\)\]）】」』>＞]\s*(?P<body>.+)$'
)
RE_SINGER_DASH = re.compile(
    r'^(?P<label>[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff/&+・ ]{1,24})\s*[-~]\s*(?P<body>.+)$'
)

TRANSLATION_BRACKETS = {
    '{}', '｛｝', '[]', '［］', '()', '（）', '「」', '『』', '【】', '〖〗', '〔〕'
}
DEFAULT_METADATA_KEYS = {
    'ti', 'ar', 'al', 'by', 'offset', 'la', 'id', 'length', 're', 've',
}
METADATA_LABELS = {
    '作词', '作詞', '词', '詞', '作曲', '曲', '编曲', '編曲', '作词作曲', '词曲',
    '演唱', '歌手', '原唱', 'vocal', 'artist', 'title', '标题', '標題', 'album',
    '专辑', '專輯', '制作', '製作', 'producer', 'lyrics', 'music', 'composer',
    'lyricist', 'chorus', '和声', '和聲', 'mix', 'mixed', 'master', '录音', '錄音',
}
SINGER_ROLE_WORDS = {
    '男', '女', '合', '和', '齐', '齊', '全', '全员', '全員', 'duet', 'solo',
    'chorus', 'rap', 'lead', 'main', 'vocal', '和声', '和聲',
}


class LyricTokenLine:
    '''
    歌词解析后的单行对象。
    '''
    def __init__(
        self,
        timestamp: float | None,
        raw_text: str,
        text: str,
        normalized_text: str,
        is_metadata: bool,
        singer: str | None = None,
        translation: str = '',
        metadata_key: str | None = None,
        metadata_value: str | None = None,
    ):
        """
        初始化单行歌词对象。

        参数:
            timestamp: 行级时间戳，单位为秒；纯文本歌词时可为 None。
            raw_text: 原始歌词文本，不做改写。
            text: 清洗后的正文文本。
            normalized_text: 适合后续对齐的规范化文本。
            is_metadata: 当前行是否被判定为元数据。
            singer: 提取出的歌手/角色标记，若无则为 None。
            translation: 与当前行对齐的翻译文本。
            metadata_key: 当行为元数据时，对应的键。
            metadata_value: 当行为元数据时，对应的值。
        """
        self.timestamp = timestamp
        self.raw_text = raw_text
        self.text = text
        self.normalized_text = normalized_text
        self.is_metadata = is_metadata
        self.singer = singer
        self.translation = translation
        self.metadata_key = metadata_key
        self.metadata_value = metadata_value


class LyricLineStamp:
    '''
    带行级时间戳的歌词对象。
    '''
    def __init__(self, lrc: str):
        """
        从 LRC 文本中解析歌词、元数据与规范化结果。

        参数:
            lrc: 原始 LRC 字符串。
        """
        self.metadata = {}
        self.metadata_keys = []
        self.line_infos = []
        self.has_real_timestamps = False

        for line in lrc.splitlines():
            line = line.strip()
            if not line:
                continue

            meta_match = RE_METADATA.match(line)
            if meta_match:
                key, value = meta_match.groups()
                self._add_metadata(key, value)
                continue

            lyric_match = RE_LYRICS.match(line)
            if not lyric_match:
                continue

            timestamp, text = lyric_match.groups()
            minute, second = timestamp.split(':')
            parsed = self._parse_lyric_line(60 * float(minute) + float(second), text)
            self.line_infos.append(parsed)
            self.has_real_timestamps = True

            if parsed.is_metadata:
                self._add_metadata(parsed.metadata_key or 'meta', parsed.metadata_value or parsed.text)
                continue

    @classmethod
    def _from_line_infos(
        cls,
        line_infos: list[LyricTokenLine],
        metadata: dict[str, str] | None = None,
        metadata_keys: list[str] | None = None,
        has_real_timestamps: bool = False,
    ):
        lyric = cls("")
        lyric.line_infos = line_infos
        lyric.metadata = dict(metadata or {})
        lyric.metadata_keys = list(metadata_keys or [])
        lyric.has_real_timestamps = has_real_timestamps
        return lyric

    @classmethod
    def from_plain_text(cls, text: str):
        """
        从不带时间戳的纯文本歌词构造歌词对象。

        参数:
            text: 按行分隔的纯文本歌词。

        说明:
            纯文本歌词只保留顺序，不再伪造时间戳。
            这样可以避免 offset-only 把“顺序信息”误当成真实时间轴。
        """
        line_infos: list[LyricTokenLine] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_infos.append(cls._parse_lyric_line(None, line))
        return cls._from_line_infos(line_infos=line_infos, has_real_timestamps=False)

    @property
    def lyric_lines(self) -> list[LyricTokenLine]:
        """
        获取所有非元数据的歌词行。

        返回:
            仅包含实际歌词行的对象列表。
        """
        return [line for line in self.line_infos if not line.is_metadata]

    @property
    def timestamps(self) -> list[float | None]:
        """
        获取歌词时间戳列表。

        返回:
            与歌词行一一对应的秒级时间戳列表；纯文本歌词时元素可为 None。
        """
        return [line.timestamp for line in self.lyric_lines]

    @property
    def lyrics(self) -> list[str]:
        """
        获取歌词正文列表。

        返回:
            清洗后的歌词正文列表。
        """
        return [line.text for line in self.lyric_lines]

    @property
    def normalized_lyrics(self) -> list[str]:
        """
        获取规范化后的歌词文本列表。

        返回:
            适合做对齐或匹配的歌词文本列表。
        """
        return [line.normalized_text for line in self.lyric_lines]

    @property
    def singers(self) -> list[str | None]:
        """
        获取每行歌词对应的歌手/角色标记。

        返回:
            与歌词行对应的歌手标记列表，未识别时为 None。
        """
        return [line.singer for line in self.lyric_lines]

    @property
    def translation(self) -> list[str]:
        """
        获取每行歌词对应的翻译文本。

        返回:
            与歌词行对应的翻译列表。
        """
        return [line.translation for line in self.lyric_lines]

    def _add_metadata(self, key: str, value: str):
        """
        保存元数据，并统一键名格式。

        参数:
            key: 元数据键。
            value: 元数据值。
        """
        key_norm = self.normalize_plain_text(key, keep_spaces=False).lower()
        value = self.normalize_plain_text(value)
        if key_norm not in self.metadata:
            self.metadata_keys.append(key_norm)
        self.metadata[key_norm] = value

    @staticmethod
    def normalize_plain_text(text: str, keep_spaces: bool = True) -> str:
        """
        做基础文本归一化，不涉及歌词特有规则。

        参数:
            text: 待处理文本。
            keep_spaces: 是否保留单个空格分词。

        返回:
            归一化后的文本。
        """
        return normalize_text(text, keep_spaces=keep_spaces)

    @classmethod
    def normalize_lyric_text(cls, text: str) -> str:
        """
        对歌词正文做规范化，便于后续对齐。

        参数:
            text: 歌词正文。

        返回:
            去掉注音和部分标点后的规范化歌词文本。
        """
        text = cls.normalize_plain_text(text)
        text = RE_FURIGANA.sub(r'\1', text)
        text = RE_LYRIC_QUOTES.sub(' ', text)
        text = re.sub(r'[·•♪♬♫]+', ' ', text)
        text = re.sub(r'[，。！？、,.!?;；:：~～…]+', ' ', text)
        text = RE_SPACES.sub(' ', text).strip()
        return text

    @classmethod
    def _looks_like_metadata_label(cls, label: str) -> bool:
        """
        判断某个短标签是否像元数据键名。

        参数:
            label: 待判断标签。

        返回:
            若像元数据键名则返回 True。
        """
        normalized = cls.normalize_plain_text(label, keep_spaces=False).lower()
        metadata_words = {item.lower() for item in METADATA_LABELS}
        return normalized in DEFAULT_METADATA_KEYS or normalized in metadata_words

    @classmethod
    def _split_key_value(cls, text: str) -> tuple[str, str] | None:
        """
        尝试把一行文本拆成 key-value 结构。

        参数:
            text: 待拆分文本。

        返回:
            成功时返回 (key, value)，失败时返回 None。
        """
        parts = RE_METADATA_SEP.split(text, maxsplit=1)
        if len(parts) != 2:
            return None
        key, value = parts[0].strip(), parts[1].strip()
        if not key or not value:
            return None
        return key, value

    @classmethod
    def _metadata_score(cls, timestamp: float | None, raw_text: str) -> tuple[int, str | None, str | None]:
        """
        用弱规则为一行文本计算元数据倾向分数。

        参数:
            timestamp: 当前行时间戳，单位为秒；纯文本歌词时可为 None。
            raw_text: 当前行文本。

        返回:
            (分数, 元数据键, 元数据值)。
        """
        text = cls.normalize_plain_text(raw_text)
        compact = cls.normalize_plain_text(raw_text, keep_spaces=False)
        if not text:
            return 0, None, None

        score = 0
        metadata_key = None
        metadata_value = None

        key_value = cls._split_key_value(text)
        if key_value is not None:
            key, value = key_value
            if cls._looks_like_metadata_label(key):
                score += 4
                metadata_key, metadata_value = key, value
            elif timestamp is not None and timestamp <= 20 and len(key) <= 12 and len(value) <= 40:
                score += 2
                metadata_key, metadata_value = key, value

        # 早期短行更容易是“作词/作曲/演唱”之类的头部信息
        if timestamp is not None and timestamp <= 20:
            score += 1
        if len(text) <= 24:
            score += 1
        if 'instrumental' in compact.lower() or 'inst' in compact.lower():
            score += 3

        separator_count = sum(ch in ':：|｜/' for ch in text)
        lyric_char_count = sum(
            ch.isalpha() or '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff'
            for ch in text
        )
        if separator_count >= 1 and lyric_char_count <= 20:
            score += 1

        return score, metadata_key, metadata_value

    @classmethod
    def _looks_like_singer_label(cls, label: str) -> bool:
        """
        判断一个短前缀是否更像歌手/角色标记而不是歌词正文。

        参数:
            label: 待判断文本。

        返回:
            若像歌手标记则返回 True。
        """
        clean = cls.normalize_plain_text(label)
        compact = cls.normalize_plain_text(label, keep_spaces=False)
        if not clean or len(clean) > 24:
            return False
        if cls._looks_like_metadata_label(clean):
            return False
        if compact.lower() in {item.lower() for item in SINGER_ROLE_WORDS}:
            return True
        if re.fullmatch(r'[A-Za-z0-9/&+]+', compact):
            return True
        if re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9/&+・ ]+', clean):
            return True
        return False

    @classmethod
    def _extract_singer_marker(cls, text: str) -> tuple[str | None, str]:
        """
        尝试从行首提取歌手/角色标记。

        参数:
            text: 歌词原文。

        返回:
            (歌手标记, 去除标记后的正文)。
        """
        for pattern in (RE_SINGER_BRACKET, RE_SINGER_COLON, RE_SINGER_DASH):
            match = pattern.match(text)
            if not match:
                continue
            label = cls.normalize_plain_text(match.group('label'))
            body = match.group('body').strip()
            if body and cls._looks_like_singer_label(label):
                return label, body
        return None, text

    @classmethod
    def _parse_lyric_line(cls, timestamp: float | None, raw_text: str) -> LyricTokenLine:
        """
        解析单行歌词，识别元数据、歌手标记与规范化文本。

        参数:
            timestamp: 行级时间戳，单位为秒；纯文本歌词时可为 None。
            raw_text: 原始行文本。

        返回:
            解析后的 LyricTokenLine 对象。
        """
        text = cls.normalize_plain_text(raw_text)
        score, metadata_key, metadata_value = cls._metadata_score(timestamp, text)
        if score >= 4:
            return LyricTokenLine(
                timestamp=timestamp,
                raw_text=raw_text,
                text=text,
                normalized_text='',
                is_metadata=True,
                metadata_key=metadata_key,
                metadata_value=metadata_value or text,
            )

        singer, body = cls._extract_singer_marker(text)
        normalized_text = cls.normalize_lyric_text(body)
        if not normalized_text and score >= 2:
            return LyricTokenLine(
                timestamp=timestamp,
                raw_text=raw_text,
                text=text,
                normalized_text='',
                is_metadata=True,
                metadata_key=metadata_key,
                metadata_value=metadata_value or text,
            )

        return LyricTokenLine(
            timestamp=timestamp,
            raw_text=raw_text,
            text=body,
            normalized_text=normalized_text,
            is_metadata=False,
            singer=singer,
        )

    def get_alignment_texts(self, drop_empty: bool = True) -> list[str]:
        '''
        获取适用于对齐的规范化歌词文本。

        参数:
            drop_empty: 是否丢弃空字符串。

        返回:
            规范化歌词文本列表。
        '''
        if drop_empty:
            return [line.normalized_text for line in self.lyric_lines if line.normalized_text]
        return self.normalized_lyrics.copy()

    def to_lrc(self, translation: bool = False, brackets: str = "【】") -> str:
        """
        将当前对象重新导出为 LRC 文本。

        参数:
            translation: 是否同时输出翻译歌词。
            brackets: 翻译歌词外层使用的括号对。

        返回:
            导出的 LRC 字符串。
        """
        lrc = []
        if(translation and not hasattr(self, "translation")):
            raise ValueError("It should load translation first")
        if(self.metadata_keys):
            for key in self.metadata_keys:
                lrc.append(f"[{key}:{self.metadata[key]}]")
        for line in self.lyric_lines:
            timestamp = line.timestamp
            if timestamp is None:
                raise ValueError("当前歌词对象缺少真实时间戳，无法直接导出为 LRC")
            text = line.text
            minute = int(timestamp // 60)
            second = timestamp % 60
            lrc_line = f"[{minute:02d}:{second:05.2f}]{text}"
            if(translation and line.translation):
                lrc_line = lrc_line + f"{brackets[0]}{line.translation}{brackets[1]}"
            lrc.append(lrc_line)
        return '\n'.join(lrc)

    def load_translation(self, translation_lrc: str):
        """
        加载翻译歌词，并按最近时间戳写回歌词行对象。

        参数:
            translation_lrc: 翻译歌词的 LRC 字符串。
        """
        lyric_lines = self.lyric_lines
        for line_info in lyric_lines:
            line_info.translation = ''
        translation_items: list[str] = []
        for line in translation_lrc.splitlines():
            line = line.strip()
            lyric_match = RE_LYRICS.match(line)
            if lyric_match:
                timestamp_str, text = lyric_match.groups()
                translation_items.append(text)
                minute, second = timestamp_str.split(':')
                trans_timestamp = 60 * float(minute) + float(second)
                if not self.has_real_timestamps:
                    continue
                closest_idx = min(
                    range(len(lyric_lines)),
                    key=lambda i: abs(float(lyric_lines[i].timestamp or 0.0) - trans_timestamp)
                )
                lyric_lines[closest_idx].translation = text
        if not self.has_real_timestamps:
            for idx, text in enumerate(translation_items):
                if idx >= len(lyric_lines):
                    break
                lyric_lines[idx].translation = text

    def clean_translation(self, brackets: set[str] = TRANSLATION_BRACKETS,
                          threshold: float = 0.8, only_detect: bool = False) -> str | None:
        '''
        检测并去除翻译歌词两端统一的括号。

        参数:
            brackets: 候选括号对集合，每个元素形如 '【】'。
            threshold: 多数判定阈值，超过该比例才视为统一括号。
            only_detect: 若为 True，则只检测不修改。

        返回:
            检测到的括号对，若未检测到则返回 None。
        '''
        trans_texts = [line.translation for line in self.lyric_lines if len(line.translation) > 1]
        if not trans_texts:
            return None
        headtails = [(t[0] + t[-1]) for t in trans_texts]
        headtails_count = Counter(headtails)
        headtails_first, freq = headtails_count.most_common(1)[0]
        if((freq < threshold * len(trans_texts)) or (headtails_first not in brackets)):
            return None
        if(only_detect):
            return headtails_first

        re_bracket = re.compile(rf"^{re.escape(headtails_first[0])}(.*){re.escape(headtails_first[1])}$")
        for line in self.lyric_lines:
            bracket_match = re_bracket.match(line.translation)
            if(bracket_match):
                line.translation = bracket_match.group(1)
        return headtails_first
