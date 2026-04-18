import gc

import librosa
import numpy as np
from scipy.signal import find_peaks
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def _clamp01(x: np.ndarray | float) -> np.ndarray | float:
    """
    将输入裁剪到 [0, 1] 区间。

    参数:
        x: 输入标量或数组。
    """
    return np.clip(x, 0.0, 1.0)


def _moving_average(y: np.ndarray, size: int) -> np.ndarray:
    """
    对一维序列做滑动平均平滑。

    参数:
        y: 输入序列。
        size: 平滑窗口大小，单位为采样点数。
    """
    size = max(int(size), 1)
    if size <= 1:
        return y
    kernel = np.ones(size, dtype=np.float32) / size
    return np.convolve(y, kernel, mode="same")


def _intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    将布尔掩码转换成连续区间。

    参数:
        mask: 一维布尔数组，True 表示激活帧。
    """
    intervals = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


class VocalSegment:
    """
    单个分片对象。
    """

    def __init__(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
        core_start: float,
        core_end: float,
        scores: dict,
        features: dict,
        samples: tuple[int, int],
        core_samples: tuple[int, int],
    ):
        """
        初始化单个分片。

        参数:
            audio: 带 padding 的片段音频。
            start: 片段起始时间，单位秒。
            end: 片段结束时间，单位秒。
            core_start: 实际用于打分的核心片段起始时间。
            core_end: 实际用于打分的核心片段结束时间。
            scores: 片段级评分字典。
            features: 片段级底层特征字典。
            samples: 带 padding 片段的采样点区间。
            core_samples: 核心片段的采样点区间。
        """
        self.audio = audio
        self.start = start
        self.end = end
        self.duration = end - start
        self.core_start = core_start
        self.core_end = core_end
        self.core_duration = core_end - core_start
        self.scores = scores
        self.features = features
        self.samples = samples
        self.core_samples = core_samples

    def to_dict(self) -> dict:
        """
        将分片对象转换为字典，方便外部脚本直接使用。
        """
        return {
            "audio": self.audio,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "core_start": self.core_start,
            "core_end": self.core_end,
            "core_duration": self.core_duration,
            "scores": self.scores,
            "features": self.features,
            "samples": self.samples,
            "core_samples": self.core_samples,
        }


class VocalSection:
    """
    第一次分片得到的长乐段对象。
    """

    def __init__(
        self,
        section_index: int,
        start: float,
        end: float,
        segment_indices: list[int],
    ):
        """
        初始化长乐段对象。

        参数:
            section_index: 长乐段序号。
            start: 长乐段开始时间，单位秒。
            end: 长乐段结束时间，单位秒。
            segment_indices: 落在该长乐段中的细分片段序号列表。
        """
        self.section_index = section_index
        self.start = start
        self.end = end
        self.duration = end - start
        self.segment_indices = segment_indices

    def to_dict(self) -> dict:
        """
        将长乐段对象转换为字典。
        """
        return {
            "section_index": self.section_index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "segment_indices": self.segment_indices,
        }


class VocalAnalysisResult:
    """
    整首歌的人声分析结果对象。
    """

    def __init__(
        self,
        music_path: str,
        sr: int,
        is_vocal: bool,
        vocal_time: float,
        main_vocal: np.ndarray,
        vocals_stereo: np.ndarray,
        activity_profile: dict,
        sections: list[VocalSection],
        segments: list[VocalSegment],
    ):
        """
        初始化分析结果。

        参数:
            music_path: 输入音频路径。
            sr: 采样率。
            is_vocal: 是否检测到明显人声。
            vocal_time: 整首歌中估计的人声时长。
            main_vocal: 主唱增强后的单声道音频。
            vocals_stereo: Demucs 分离后的人声双声道。
            activity_profile: 帧级分析特征。
            sections: 第一次分片得到的长乐段列表。
            segments: 分片结果列表。
        """
        self.music_path = music_path
        self.sr = sr
        self.is_vocal = is_vocal
        self.vocal_time = vocal_time
        self.main_vocal = main_vocal
        self.vocals_stereo = vocals_stereo
        self.activity_profile = activity_profile
        self.sections = sections
        self.segments = segments

    def to_dict(self) -> dict:
        """
        将分析结果转换为字典。
        """
        return {
            "music_path": self.music_path,
            "sr": self.sr,
            "is_vocal": self.is_vocal,
            "vocal_time": self.vocal_time,
            "main_vocal": self.main_vocal,
            "vocals_stereo": self.vocals_stereo,
            "activity_profile": self.activity_profile,
            "sections": [section.to_dict() for section in self.sections],
            "segments": [segment.to_dict() for segment in self.segments],
        }


class VocalAnalyzer:
    """
    人声分离、分片与风险评估器。
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 256,
        smoothing_sec: float = 0.10,
        padding_sec: float = 0.18,
        enter_threshold: float = 0.52,
        keep_threshold: float = 0.30,
        min_segment_sec: float = 0.45,
        max_gap_sec: float = 0.22,
        max_segment_sec: float = 8.0,
        long_valley_threshold: float = 0.30,
        low_memory_mode: bool = False,
    ):
        """
        初始化分析器。

        参数:
            frame_length: 特征计算窗口长度。
            hop_length: 特征计算帧移。
            smoothing_sec: activity 平滑窗口时长。
            padding_sec: 输出片段首尾补边时长。
            enter_threshold: 双阈值滞回高阈值。
            keep_threshold: 双阈值滞回低阈值。
            min_segment_sec: 最小片段时长。
            max_gap_sec: 允许合并的短停顿时长。
            max_segment_sec: 最大片段时长。
            long_valley_threshold: 长片段二次切分的谷值阈值。
            low_memory_mode: 是否启用低显存模式；启用后建议在人声分析阶段结束后主动卸载 Demucs。
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.smoothing_sec = smoothing_sec
        self.padding_sec = padding_sec
        self.enter_threshold = enter_threshold
        self.keep_threshold = keep_threshold
        self.min_segment_sec = min_segment_sec
        self.max_gap_sec = max_gap_sec
        self.max_segment_sec = max_segment_sec
        self.long_valley_threshold = long_valley_threshold
        self.low_memory_mode = low_memory_mode

    @staticmethod
    def get_model():
        """
        懒加载 Demucs 模型并复用全局实例。
        """
        global _model
        if _model is None:
            _model = get_model("htdemucs")
            _model.to(device)
            _model.eval()
        return _model

    @staticmethod
    def release_model():
        """
        释放全局缓存的 Demucs 模型，避免与后续 ASR 模型同时占用显存。

        说明:
            当前分离流程使用了全局缓存模型以便重复调用时更快，
            但在“先分离、再转写”的链路里，这会导致 Demucs 和 Whisper 同时驻留在 GPU。
        """
        global _model
        if _model is None:
            return
        try:
            _model.cpu()
        except Exception:
            pass
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model_if_needed(self):
        """
        按当前模式决定是否释放 Demucs 模型。
        """
        if self.low_memory_mode:
            self.release_model()

    @staticmethod
    def ensure_stereo_44k(waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        """
        将输入音频统一到 44.1kHz 双声道。

        参数:
            waveform: 音频波形，形状为 `(C, T)`。
            sr: 原始采样率。
        """
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
            sr = 44100
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        return waveform, sr

    @staticmethod
    def extract_main_vocal(vocals_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 Demucs 人声 stem 中提取主唱增强信号。

        参数:
            vocals_np: 人声双声道波形，形状为 `(2, T)`。

        说明:
            `mid = (L + R) / 2` 更接近中置主唱，
            `side = (L - R) / 2` 更容易包含和声与空间效果。
            这里用 `mid - abs(side)` 提取更接近主唱的单声道信号。
        """
        left = vocals_np[0]
        right = vocals_np[1]
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        main_vocal = mid - np.abs(side)
        return main_vocal.astype(np.float32), mid.astype(np.float32), side.astype(np.float32)

    def load_audio_file(self, music_path: str) -> tuple[torch.Tensor, int]:
        """
        读取音频文件并统一格式。

        参数:
            music_path: 音频文件路径。
        """
        waveform, sr = torchaudio.load(music_path)
        return self.ensure_stereo_44k(waveform, sr)

    def separate_vocals_file(self, music_path: str) -> dict:
        """
        使用 Demucs 分离人声，并返回后续分析所需的中间结果。

        参数:
            music_path: 音频文件路径。
        """
        waveform, sr = self.load_audio_file(music_path)
        reference_mono = waveform.mean(dim=0).cpu().numpy()
        model = self.get_model()

        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0).to(device),
                device=device,
                progress=False,
                split=True,
                overlap=0.25,
            )

        vocals = sources[0, model.sources.index("vocals")].detach().cpu().numpy()
        main_vocal, mid, side = self.extract_main_vocal(vocals)

        del sources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "sr": sr,
            "waveform": waveform.cpu().numpy(),
            "reference_mono": reference_mono.astype(np.float32),
            "vocals_stereo": vocals.astype(np.float32),
            "vocals_mono": vocals.mean(axis=0).astype(np.float32),
            "main_vocal": main_vocal,
            "mid": mid,
            "side": side,
        }

    @staticmethod
    def has_vocals_from_array(
        y: np.ndarray,
        sr: int = 44100,
        origin: np.ndarray | None = None,
        rms_db_threshold: float = -30,
        min_duration_sec: float = 5.0,
        min_relative_vocal_time: float = 0.2,
    ) -> tuple[bool, float]:
        """
        根据 RMS 能量粗略判断整段音频是否包含足够多人声。

        参数:
            y: 单声道人声波形。
            sr: 采样率。
            origin: 原始参考音频，用于确定分贝参考值。
            rms_db_threshold: 判定为有人声的 RMS 分贝阈值。
            min_duration_sec: 最少人声时长。
            min_relative_vocal_time: 人声时长占整段的最小比例。
        """
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if origin is None:
            ref = 1.0
        else:
            ref = max(float(np.max(np.abs(origin))), 1e-6)
        rms_db = librosa.amplitude_to_db(rms, ref=ref)
        voiced = rms_db > rms_db_threshold
        voiced_time = voiced.sum() * 512 / sr
        song_time = rms_db.shape[0] * 512 / sr
        is_vocal = (voiced_time >= min_duration_sec) and (voiced_time >= min_relative_vocal_time * song_time)
        return is_vocal, voiced_time

    def build_activity_profile(
        self,
        main_vocal: np.ndarray,
        mid: np.ndarray,
        side: np.ndarray,
        sr: int,
    ) -> dict:
        """
        构造帧级 activity 曲线和 center 曲线。

        参数:
            main_vocal: 主唱增强后的单声道波形。
            mid: 中置信号。
            side: 侧置信号。
            sr: 采样率。

        说明:
            `activity` 负责分句，核心思想是“能量门控 + 中置信号校正”。
            `center_score_long` 则服务于后续片段风险打分，用于判断片段是否持续偏离中置。
        """
        main_rms = librosa.feature.rms(
            y=main_vocal,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        mid_rms = librosa.feature.rms(
            y=mid,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        side_rms = librosa.feature.rms(
            y=np.abs(side),
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        ref = max(float(np.max(main_rms)), 1e-6)
        main_db = librosa.amplitude_to_db(main_rms, ref=ref)
        db_floor = float(np.percentile(main_db, 5))
        db_ceil = float(np.percentile(main_db, 95))
        db_scale = max(db_ceil - db_floor, 1.0)
        energy_score = _clamp01((main_db - db_floor) / db_scale)

        center_ratio = mid_rms / (mid_rms + side_rms + 1e-8)
        center_floor = float(np.percentile(center_ratio, 10))
        center_ceil = float(np.percentile(center_ratio, 90))
        center_scale = max(center_ceil - center_floor, 1e-3)
        center_score = _clamp01((center_ratio - center_floor) / center_scale)

        center_smooth_size = max(int(round(max(self.smoothing_sec * 2.5, 0.18) * sr / self.hop_length)), 1)
        center_score_smooth = _clamp01(_moving_average(center_score, center_smooth_size))
        center_long_size = max(int(round(max(0.85, self.smoothing_sec * 10.0) * sr / self.hop_length)), 1)
        center_score_long = _clamp01(_moving_average(center_score, center_long_size))

        # activity 用于第一次分片与二次断句。
        # 能量决定“有没有唱”，center_smooth 负责压制明显偏离中置的背景人声。
        activity = energy_score * (0.35 + 0.65 * center_score_smooth)
        smooth_size = max(int(round(self.smoothing_sec * sr / self.hop_length)), 1)
        activity_smooth = _moving_average(activity, smooth_size)
        split_smooth_size = max(int(round(max(0.30, self.smoothing_sec * 4.0) * sr / self.hop_length)), 1)
        split_activity = _moving_average(activity_smooth, split_smooth_size)
        frame_times = librosa.frames_to_time(np.arange(len(activity_smooth)), sr=sr, hop_length=self.hop_length)

        return {
            "main_rms": main_rms,
            "mid_rms": mid_rms,
            "side_rms": side_rms,
            "main_db": main_db,
            "energy_score": energy_score,
            "center_ratio": center_ratio,
            "center_score": center_score,
            "center_score_smooth": center_score_smooth,
            "center_score_long": center_score_long,
            "activity": activity_smooth,
            "split_activity": split_activity,
            "frame_times": frame_times,
            "db_floor": db_floor,
            "db_ceil": db_ceil,
            "db_scale": db_scale,
            "center_floor": center_floor,
            "center_ceil": center_ceil,
        }

    def split_long_interval(
        self,
        start_frame: int,
        end_frame: int,
        activity: np.ndarray,
        split_activity: np.ndarray,
        sr: int,
    ) -> list[tuple[int, int]]:
        """
        对过长区间做二次断句。

        参数:
            start_frame: 区间起始帧。
            end_frame: 区间结束帧。
            activity: 原始 activity 曲线。
            split_activity: 更平滑的断句曲线。
            sr: 采样率。

        说明:
            第一次滞回分片往往得到大段级区间。
            这里在平滑后的 `split_activity` 上寻找谷底峰值，只在找到足够显著的谷底时切分，
            否则退回最大片段时长硬切，避免区间无限增长。
        """
        intervals = [(start_frame, end_frame)]
        min_frames = max(int(round(self.min_segment_sec * sr / self.hop_length)), 1)
        max_frames = max(int(round(self.max_segment_sec * sr / self.hop_length)), min_frames + 1)
        valley_window = max(int(round(0.18 * sr / self.hop_length)), 1)
        min_peak_distance = max(min_frames // 2, 1)

        while True:
            next_intervals = []
            changed = False
            for cur_start, cur_end in intervals:
                if cur_end - cur_start <= max_frames:
                    next_intervals.append((cur_start, cur_end))
                    continue

                local_split = split_activity[cur_start:cur_end]
                if local_split.size <= 2 * min_frames:
                    next_intervals.append((cur_start, cur_end))
                    continue

                search_local = local_split[min_frames: len(local_split) - min_frames]
                best_split = None
                best_score = float("-inf")

                if search_local.size > 0:
                    dynamic_range = float(np.percentile(local_split, 85) - np.percentile(local_split, 15))
                    min_prominence = max(0.035, 0.22 * dynamic_range)
                    valley_indices, valley_props = find_peaks(
                        -search_local,
                        prominence=min_prominence,
                        distance=min_peak_distance,
                    )

                    if valley_indices.size > 0:
                        prominences = valley_props.get("prominences", np.zeros_like(valley_indices, dtype=np.float32))
                        interval_median = float(np.median(local_split))
                        interval_low = float(np.percentile(local_split, 20))

                        for local_idx, prominence in zip(valley_indices, prominences):
                            frame_idx = cur_start + min_frames + int(local_idx)
                            local_start = max(cur_start, frame_idx - valley_window)
                            local_end = min(cur_end, frame_idx + valley_window)

                            valley_value = float(np.mean(split_activity[local_start:local_end]))
                            raw_value = float(np.mean(activity[local_start:local_end]))
                            valley_depth = interval_median - valley_value
                            low_bonus = max(0.0, interval_low - valley_value)
                            distance_penalty = abs(frame_idx - (cur_start + cur_end) / 2) / max(cur_end - cur_start, 1)
                            raw_bonus = max(0.0, interval_median - raw_value)

                            candidate_score = (
                                1.35 * float(prominence)
                                + 0.55 * valley_depth
                                + 0.35 * low_bonus
                                + 0.20 * raw_bonus
                                - 0.18 * distance_penalty
                            )
                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_split = frame_idx

                if best_split is None or best_score < self.long_valley_threshold:
                    hard_split = cur_start + max_frames
                    if cur_end - hard_split < min_frames:
                        hard_split = (cur_start + cur_end) // 2
                    best_split = hard_split

                next_intervals.append((cur_start, best_split))
                next_intervals.append((best_split, cur_end))
                changed = True

            intervals = next_intervals
            if not changed:
                break

        return intervals

    def _initial_segment_intervals(self, activity: np.ndarray, sr: int) -> list[tuple[int, int]]:
        """
        先用双阈值和短停顿合并得到长乐段级区间。

        参数:
            activity: 帧级 activity 曲线。
            sr: 采样率。
        """
        high_mask = activity >= self.enter_threshold
        low_mask = activity >= self.keep_threshold
        candidate_intervals = []

        for high_start, high_end in _intervals_from_mask(high_mask):
            left = high_start
            right = high_end
            while left > 0 and low_mask[left - 1]:
                left -= 1
            while right < len(low_mask) and low_mask[right]:
                right += 1
            candidate_intervals.append((left, right))

        if not candidate_intervals:
            return []

        merged = [candidate_intervals[0]]
        max_gap_frames = int(round(self.max_gap_sec * sr / self.hop_length))
        for start, end in candidate_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap_frames:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        min_frames = int(round(self.min_segment_sec * sr / self.hop_length))
        return [(start, end) for start, end in merged if end - start >= max(min_frames, 1)]

    def segment_activity_frames(
        self,
        activity: np.ndarray,
        split_activity: np.ndarray,
        sr: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        根据 activity 曲线生成长乐段区间和细分分片区间。

        参数:
            activity: 帧级 activity 曲线。
            split_activity: 更平滑的断句曲线。
            sr: 采样率。

        说明:
            先保留第一次分片得到的长乐段，再对每个长乐段做二次断句。
        """
        coarse_intervals = self._initial_segment_intervals(activity=activity, sr=sr)
        refined = []
        for start, end in coarse_intervals:
            refined.extend(self.split_long_interval(start, end, activity, split_activity, sr))
        return coarse_intervals, refined

    def score_segment(
        self,
        main_vocal: np.ndarray,
        sr: int,
        start_sample: int,
        end_sample: int,
        activity_profile: dict,
    ) -> VocalSegment:
        """
        对单个分片计算分数与底层特征。

        参数:
            main_vocal: 主唱增强后的单声道波形。
            sr: 采样率。
            start_sample: 片段起始采样点。
            end_sample: 片段结束采样点。
            activity_profile: `build_activity_profile` 返回的帧级分析结果。

        说明:
            最终只保留两个分数：
            1. `vocal_presence` 用于判断这段是否值得送转写。
            2. `off_center_risk` 用于评估该段偏离中置的程度，可作为转写前筛选与置信度降权依据。
        """
        pad_samples = int(round(self.padding_sec * sr))
        total_samples = len(main_vocal)
        core_start = max(0, start_sample)
        core_end = min(total_samples, end_sample)
        start = max(0, start_sample - pad_samples)
        end = min(total_samples, end_sample + pad_samples)

        audio = main_vocal[start:end]

        start_frame = max(int(core_start / self.hop_length), 0)
        end_frame = min(int(np.ceil(core_end / self.hop_length)), len(activity_profile["activity"]))
        frame_slice = slice(start_frame, max(start_frame + 1, end_frame))

        activity_mean = float(np.mean(activity_profile["activity"][frame_slice]))
        energy_mean = float(np.mean(activity_profile["energy_score"][frame_slice]))
        center_long_frames = activity_profile["center_score_long"][frame_slice]
        center_long_q80 = float(np.percentile(center_long_frames, 80))
        center_low_ratio = float(np.mean(center_long_frames < 0.35))
        vocal_presence_score = float(_clamp01(0.62 * activity_mean + 0.38 * energy_mean))
        off_center_core = float(
            _clamp01(
                0.62 * (1.0 - center_long_q80)
                + 0.38 * center_low_ratio
            )
        )
        off_center_risk_score = float(vocal_presence_score * off_center_core)

        return VocalSegment(
            audio=audio.astype(np.float32),
            start=start / sr,
            end=end / sr,
            core_start=core_start / sr,
            core_end=core_end / sr,
            scores={
                "vocal_presence": vocal_presence_score,
                "off_center_risk": off_center_risk_score,
            },
            features={
                "activity_mean": activity_mean,
                "energy_mean": energy_mean,
                "center_long_q80": center_long_q80,
                "center_low_ratio": center_low_ratio,
            },
            samples=(start, end),
            core_samples=(core_start, core_end),
        )

    def segment_vocals_soft(self, vocals: np.ndarray, sr: int) -> tuple[list[VocalSection], list[VocalSegment], dict]:
        """
        对分离后的人声做细粒度切片，并计算每个片段的分数。

        参数:
            vocals: 人声波形，形状为 `(2, T)` 或 `(T,)`。
            sr: 采样率。
        """
        if vocals.ndim == 1:
            main_vocal = vocals.astype(np.float32)
            mid = main_vocal
            side = np.zeros_like(main_vocal)
        else:
            main_vocal, mid, side = self.extract_main_vocal(vocals)

        activity_profile = self.build_activity_profile(main_vocal=main_vocal, mid=mid, side=side, sr=sr)
        coarse_intervals, _ = self.segment_activity_frames(
            activity=activity_profile["activity"],
            split_activity=activity_profile["split_activity"],
            sr=sr,
        )

        sections = []
        segments = []
        for section_index, (coarse_start, coarse_end) in enumerate(coarse_intervals, start=1):
            section_segment_indices = []
            refined_intervals = self.split_long_interval(
                coarse_start,
                coarse_end,
                activity_profile["activity"],
                activity_profile["split_activity"],
                sr,
            )
            for start_frame, end_frame in refined_intervals:
                start_sample = int(start_frame * self.hop_length)
                end_sample = int(end_frame * self.hop_length)
                segment = self.score_segment(
                    main_vocal=main_vocal,
                    sr=sr,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    activity_profile=activity_profile,
                )
                if segment.duration < self.min_segment_sec:
                    continue
                segments.append(segment)
                section_segment_indices.append(len(segments))

            if not section_segment_indices:
                continue

            sections.append(
                VocalSection(
                    section_index=section_index,
                    start=coarse_start * self.hop_length / sr,
                    end=coarse_end * self.hop_length / sr,
                    segment_indices=section_segment_indices,
                )
            )

        return sections, segments, activity_profile

    def analyze_file(self, music_path: str) -> VocalAnalysisResult:
        """
        对音频文件执行完整的人声分析流程。

        参数:
            music_path: 音频文件路径。
        """
        separated = self.separate_vocals_file(music_path)
        sections, segments, activity_profile = self.segment_vocals_soft(separated["vocals_stereo"], sr=separated["sr"])
        is_vocal, vocal_time = self.has_vocals_from_array(
            separated["main_vocal"],
            sr=separated["sr"],
            origin=separated["reference_mono"],
        )
        return VocalAnalysisResult(
            music_path=music_path,
            sr=separated["sr"],
            is_vocal=is_vocal,
            vocal_time=vocal_time,
            main_vocal=separated["main_vocal"],
            vocals_stereo=separated["vocals_stereo"],
            activity_profile=activity_profile,
            sections=sections,
            segments=segments,
        )
