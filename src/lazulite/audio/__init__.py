"""音频 I/O、分离与活动分析。"""

from .io import decode_audio, read_audio_metadata, resample_audio
from .activity import VocalConfig, analyze_vocals
from .bs_karaoke import BSKaraokeConfig, BSKaraokeSeparator
from .separation import DemucsConfig, DemucsSeparator

__all__ = ["BSKaraokeConfig", "BSKaraokeSeparator", "DemucsConfig", "DemucsSeparator", "VocalConfig", "analyze_vocals", "decode_audio", "read_audio_metadata", "resample_audio"]
