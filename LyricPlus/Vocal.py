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
    x: 输入标量或数组
    """
    return np.clip(x, 0.0, 1.0)


def _get_model():
    """
    懒加载 Demucs 模型并复用全局实例。
    """
    global _model
    if _model is None:
        _model = get_model("htdemucs")
        _model.to(device)
        _model.eval()
    return _model


def _ensure_stereo_44k(waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    """
    将输入音频统一为 44.1kHz 双声道。

    参数:
    waveform: `torchaudio.load` 读出的波形，形状通常为 `(C, T)`
    sr: 原始采样率
    """
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
        sr = 44100
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform, sr


def _moving_average(y: np.ndarray, size: int) -> np.ndarray:
    """
    对一维序列做滑动平均平滑。

    参数:
    y: 输入序列
    size: 平滑窗口大小，单位为采样点数
    """
    size = max(int(size), 1)
    if size <= 1:
        return y
    kernel = np.ones(size, dtype=np.float32) / size
    return np.convolve(y, kernel, mode="same")


def _intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    将布尔掩码转成连续区间列表。

    参数:
    mask: 一维布尔数组，True 表示激活帧
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


def extract_main_vocal(vocals_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 Demucs 输出的人声 stem 中提取主唱增强信号。

    参数:
    vocals_np: 人声双声道波形，形状为 `(2, T)`

    说明:
    `mid = (L + R) / 2` 更接近居中的主唱，
    `side = (L - R) / 2` 更容易包含左右铺开的和声或空间效果，
    这里用 `mid - abs(side)` 来压低侧声道成分，得到更适合后续分析的主唱增强信号。
    """
    left = vocals_np[0]
    right = vocals_np[1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    main_vocal = mid - np.abs(side)
    return main_vocal.astype(np.float32), mid.astype(np.float32), side.astype(np.float32)


def load_audio_file(music_path: str) -> tuple[torch.Tensor, int]:
    """
    读取音频文件并统一格式。

    参数:
    music_path: 音频文件路径
    """
    waveform, sr = torchaudio.load(music_path)
    return _ensure_stereo_44k(waveform, sr)


def separate_vocals_file(music_path: str) -> dict:
    """
    使用 Demucs 对整首歌做人声分离，并返回后续分析需要的中间结果。

    参数:
    music_path: 音频文件路径
    """
    waveform, sr = load_audio_file(music_path)
    reference_mono = waveform.mean(dim=0).cpu().numpy()
    model = _get_model()

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
    main_vocal, mid, side = extract_main_vocal(vocals)

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


def has_vocals_from_array(
    y: np.ndarray,
    sr: int = 44100,
    origin: np.ndarray | None = None,
    rms_db_threshold: float = -30,
    min_duration_sec: float = 5.0,
    min_relative_vocal_time: float = 0.2
) -> tuple[bool, float]:
    """
    根据 RMS 能量粗略判断一段音频是否包含足够多的人声。

    参数:
    y: 单声道音频波形
    sr: 采样率
    origin: 原始参考音频，用于估算相对分贝参考值
    rms_db_threshold: 判定为有人声的 RMS 分贝阈值
    min_duration_sec: 最少人声时长
    min_relative_vocal_time: 人声时长占整段音频的最小比例
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512
    )[0]

    if(origin is None):
        ref = 1.0
    else:
        ref = max(float(np.max(np.abs(origin))), 1e-6)
    rms_db = librosa.amplitude_to_db(rms, ref=ref)

    voiced = rms_db > rms_db_threshold
    voiced_time = voiced.sum() * 512 / sr
    song_time = rms_db.shape[0] * 512 / sr

    is_vocal = (voiced_time >= min_duration_sec) and (voiced_time >= min_relative_vocal_time * song_time)
    return is_vocal, voiced_time


def detect_vocals_file(music_path: str) -> tuple[bool, float]:
    """
    对音频文件做整首级别人声检测。

    参数:
    music_path: 音频文件路径
    """
    separated = separate_vocals_file(music_path)
    return has_vocals_from_array(
        separated["main_vocal"],
        sr=separated["sr"],
        origin=separated["reference_mono"],
    )


def _build_activity_profile(
    main_vocal: np.ndarray,
    mid: np.ndarray,
    side: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    smoothing_sec: float,
) -> dict:
    """
    构造帧级 activity 曲线，供后续切片使用。

    参数:
    main_vocal: 主唱增强后的单声道波形
    mid: 中置信号
    side: 侧置信号
    sr: 采样率
    frame_length: RMS 分析窗长度
    hop_length: 帧移
    smoothing_sec: 对 activity 平滑时使用的时间窗口，单位秒

    说明:
    这里不直接用纯能量切片，而是把“能量强度”和“居中程度”混合成 activity。
    对日语流行曲来说，主唱通常更中置，而和声、混响、铺底更容易出现在 side 中。
    """
    main_rms = librosa.feature.rms(y=main_vocal, frame_length=frame_length, hop_length=hop_length)[0]
    mid_rms = librosa.feature.rms(y=mid, frame_length=frame_length, hop_length=hop_length)[0]
    side_rms = librosa.feature.rms(y=np.abs(side), frame_length=frame_length, hop_length=hop_length)[0]

    ref = max(float(np.max(main_rms)), 1e-6)
    main_db = librosa.amplitude_to_db(main_rms, ref=ref)

    # 使用分位数线性归一化保留能量的相对起伏。
    # 这样能更好地保留句内谷底与尾声弱峰之间的差别，不会像 sigmoid 那样把中低能量整体抬高。
    db_floor = float(np.percentile(main_db, 5))
    db_ceil = float(np.percentile(main_db, 95))
    db_scale = max(db_ceil - db_floor, 1.0)
    energy_score = _clamp01((main_db - db_floor) / db_scale)

    center_ratio = mid_rms / (mid_rms + side_rms + 1e-8)
    center_floor = float(np.percentile(center_ratio, 10))
    center_ceil = float(np.percentile(center_ratio, 90))
    center_scale = max(center_ceil - center_floor, 1e-3)
    center_score = _clamp01((center_ratio - center_floor) / center_scale)
    center_smooth_size = max(int(round(max(smoothing_sec * 2.5, 0.18) * sr / hop_length)), 1)
    center_score_smooth = _moving_average(center_score, center_smooth_size)
    center_score_smooth = _clamp01(center_score_smooth)

    # activity 由能量门控主导，center_score 只作为修正项。
    # 这样一来，只有在能量足够高时，居中度才会帮助提升 activity；
    # 低能量区域即使较为中置，也不会被整体“托”到高位。
    activity = energy_score * (0.35 + 0.65 * center_score_smooth)

    # 轻微平滑，减少瞬时抖动导致的过碎分片。
    smooth_size = max(int(round(smoothing_sec * sr / hop_length)), 1)
    activity_smooth = _moving_average(activity, smooth_size)
    split_smooth_size = max(int(round(max(0.30, smoothing_sec * 4.0) * sr / hop_length)), 1)
    split_activity = _moving_average(activity_smooth, split_smooth_size)
    frame_times = librosa.frames_to_time(np.arange(len(activity_smooth)), sr=sr, hop_length=hop_length)

    return {
        "main_rms": main_rms,
        "mid_rms": mid_rms,
        "side_rms": side_rms,
        "main_db": main_db,
        "energy_score": energy_score,
        "center_ratio": center_ratio,
        "center_score": center_score,
        "center_score_smooth": center_score_smooth,
        "activity": activity_smooth,
        "split_activity": split_activity,
        "frame_times": frame_times,
        "db_floor": db_floor,
        "db_ceil": db_ceil,
        "db_scale": db_scale,
        "center_floor": center_floor,
        "center_ceil": center_ceil,
    }


def _split_long_interval(
    start_frame: int,
    end_frame: int,
    activity: np.ndarray,
    split_activity: np.ndarray,
    sr: int,
    hop_length: int,
    max_segment_sec: float,
    valley_threshold: float,
    min_segment_sec: float,
) -> list[tuple[int, int]]:
    """
    将过长的连续片段按 activity 谷值继续切细。

    参数:
    start_frame: 起始帧
    end_frame: 结束帧
    activity: 帧级 activity 曲线
    split_activity: 更平滑的低频曲线，专门用于寻找句间谷底
    sr: 采样率
    hop_length: 帧移
    max_segment_sec: 允许的最大片段时长
    valley_threshold: 谷值阈值，越小越倾向于只在明显停顿处切开
    min_segment_sec: 最小片段时长

    说明:
    第一轮 activity 检测后仍可能得到“大段级”区间。
    这里会在区间内部寻找较低 activity 的局部谷值作为切点，
    尽量把片段切到更接近一句歌词的粒度。
    """
    intervals = [(start_frame, end_frame)]
    min_frames = max(int(round(min_segment_sec * sr / hop_length)), 1)
    max_frames = max(int(round(max_segment_sec * sr / hop_length)), min_frames + 1)
    valley_window = max(int(round(0.18 * sr / hop_length)), 1)
    min_peak_distance = max(min_frames // 2, 1)

    while True:
        next_intervals = []
        changed = False
        for cur_start, cur_end in intervals:
            if cur_end - cur_start <= max_frames:
                next_intervals.append((cur_start, cur_end))
                continue

            local = split_activity[cur_start:cur_end]
            if local.size <= 2 * min_frames:
                next_intervals.append((cur_start, cur_end))
                continue

            search_start = cur_start + min_frames
            search_end = cur_end - min_frames
            local_split = split_activity[cur_start:cur_end]
            local_raw = activity[cur_start:cur_end]
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

                        # 综合考虑谷底显著性(prominence)、谷深、以及不要过分偏向边缘。
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

            # 如果没有足够显著的谷底，就退化成按长度硬切，避免片段过长。
            if best_split is None or best_score < valley_threshold:
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


def _segment_activity_frames(
    activity: np.ndarray,
    split_activity: np.ndarray,
    sr: int,
    hop_length: int,
    enter_threshold: float,
    keep_threshold: float,
    min_segment_sec: float,
    max_gap_sec: float,
    max_segment_sec: float,
    long_valley_threshold: float,
) -> list[tuple[int, int]]:
    """
    根据 activity 曲线生成初始分片区间。

    参数:
    activity: 帧级 activity 曲线
    split_activity: 更平滑的低频曲线，主要用于长片段二次切分
    sr: 采样率
    hop_length: 帧移
    enter_threshold: 进入激活状态的高阈值
    keep_threshold: 保持激活状态的低阈值
    min_segment_sec: 最小片段时长
    max_gap_sec: 允许合并的最大短停顿
    max_segment_sec: 最大片段时长
    long_valley_threshold: 长片段二次切分时的谷值阈值

    说明:
    这里使用“双阈值滞回”来做分片:
    先用高阈值找到可靠核心，再用低阈值向两边扩张。
    这样比单一阈值更稳，不容易被尾音、呼吸声或轻微伴唱抖动切碎。
    """
    high_mask = activity >= enter_threshold
    low_mask = activity >= keep_threshold
    candidate_intervals = []

    for high_start, high_end in _intervals_from_mask(high_mask):
        # 先找到高置信核心，再用低阈值向两边扩张，把一句歌词的弱起弱收包进去。
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
    max_gap_frames = int(round(max_gap_sec * sr / hop_length))
    for start, end in candidate_intervals[1:]:
        prev_start, prev_end = merged[-1]
        # 将非常短的停顿合并，避免一句歌词被切成两三段。
        if start - prev_end <= max_gap_frames:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    min_frames = int(round(min_segment_sec * sr / hop_length))
    filtered = [(start, end) for start, end in merged if end - start >= max(min_frames, 1)]

    refined = []
    for start, end in filtered:
        refined.extend(
            _split_long_interval(
                start,
                end,
                activity,
                split_activity,
                sr=sr,
                hop_length=hop_length,
                max_segment_sec=max_segment_sec,
                valley_threshold=long_valley_threshold,
                min_segment_sec=min_segment_sec,
            )
        )

    return refined


def _safe_pitch_features(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    估计一段音频的有声比例与音高稳定度。

    参数:
    y: 单声道音频波形
    sr: 采样率

    说明:
    使用 `librosa.pyin` 抽取基频。
    这里不追求精确音高，而是用“是否持续有可追踪基频”和“音高变化是否平滑”作为主唱质量信号。
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and voiced_flag.size else 0.0
        if f0 is None:
            return voiced_ratio, 0.0
        valid_f0 = f0[np.isfinite(f0)]
        if valid_f0.size < 3:
            return voiced_ratio, 0.0
        midi = librosa.hz_to_midi(valid_f0)
        pitch_delta = np.diff(midi)
        pitch_stability = float(np.exp(-np.std(pitch_delta) / 2.5))
        return voiced_ratio, float(_clamp01(pitch_stability))
    except Exception:
        return 0.0, 0.0


def _score_segment(
    main_vocal: np.ndarray,
    mid: np.ndarray,
    side: np.ndarray,
    sr: int,
    start_sample: int,
    end_sample: int,
    activity_profile: dict,
    hop_length: int,
    padding_sec: float,
) -> dict:
    """
    对单个片段计算软分数与辅助特征。

    参数:
    main_vocal: 主唱增强后的单声道波形
    mid: 中置信号
    side: 侧置信号
    sr: 采样率
    start_sample: 片段起始采样点
    end_sample: 片段结束采样点
    activity_profile: `_build_activity_profile` 输出的帧级特征
    hop_length: 帧移
    padding_sec: 片段首尾补边时长，单位秒

    说明:
    这里不做硬分类，而是输出多个软分数:
    - `vocal_presence`: 这段像不像“有人在唱”
    - `lead_vocal`: 这段更像主唱还是边缘/和声
    - `harmony_risk`: 这段受和声、叠唱、空间扩散影响的风险
    - `usable`: 综合可用性，供后续 ASR 或对齐时排序/降权
    """
    pad_samples = int(round(padding_sec * sr))
    total_samples = len(main_vocal)
    start = max(0, start_sample - pad_samples)
    end = min(total_samples, end_sample + pad_samples)

    audio = main_vocal[start:end]
    audio_mid = mid[start:end]
    audio_side = side[start:end]
    duration = (end - start) / sr

    start_frame = max(int(start / hop_length), 0)
    end_frame = min(int(np.ceil(end / hop_length)), len(activity_profile["activity"]))
    frame_slice = slice(start_frame, max(start_frame + 1, end_frame))

    activity_mean = float(np.mean(activity_profile["activity"][frame_slice]))
    energy_mean = float(np.mean(activity_profile["energy_score"][frame_slice]))
    center_mean = float(np.mean(activity_profile["center_ratio"][frame_slice]))

    # 频谱平坦度越低，通常越像有明确谐波结构的人声，而不是噪声或空气感残响。
    flatness = librosa.feature.spectral_flatness(y=audio + 1e-8)[0]
    flatness_mean = float(np.mean(flatness)) if flatness.size else 1.0
    flatness_score = float(1.0 - _clamp01(flatness_mean / 0.35))

    # 过高的过零率通常意味着噪声感更强，不利于稳定转写。
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=256)[0]
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    zcr_score = float(1.0 - _clamp01((zcr_mean - 0.04) / 0.12))

    side_energy = float(np.sqrt(np.mean(audio_side ** 2)) + 1e-8)
    mid_energy = float(np.sqrt(np.mean(audio_mid ** 2)) + 1e-8)
    center_energy_ratio = float(mid_energy / (mid_energy + side_energy))

    voiced_ratio, pitch_stability = _safe_pitch_features(audio, sr)

    # presence 更偏“有没有唱”，lead 更偏“像不像稳定主唱”，harmony_risk 更偏“多声部干扰风险”。
    vocal_presence_score = float(_clamp01(0.55 * activity_mean + 0.30 * energy_mean + 0.15 * voiced_ratio))
    lead_vocal_score = float(
        _clamp01(
            0.40 * center_energy_ratio
            + 0.25 * center_mean
            + 0.20 * pitch_stability
            + 0.10 * flatness_score
            + 0.05 * zcr_score
        )
    )
    harmony_risk_score = float(
        _clamp01(
            0.45 * (1.0 - center_energy_ratio)
            + 0.30 * (1.0 - pitch_stability)
            + 0.15 * (1.0 - flatness_score)
            + 0.10 * (1.0 - zcr_score)
        )
    )
    # usable 不是硬判定，而是给下游使用的综合排序分数。
    usable_score = float(
        _clamp01(
            0.55 * vocal_presence_score
            + 0.35 * lead_vocal_score
            - 0.25 * harmony_risk_score
        )
    )

    return {
        "audio": audio.astype(np.float32),
        "start": start / sr,
        "end": end / sr,
        "duration": duration,
        "samples": (start, end),
        "scores": {
            "vocal_presence": vocal_presence_score,
            "lead_vocal": lead_vocal_score,
            "harmony_risk": harmony_risk_score,
            "usable": usable_score,
        },
        "features": {
            "activity_mean": activity_mean,
            "energy_mean": energy_mean,
            "center_ratio": center_mean,
            "center_energy_ratio": center_energy_ratio,
            "voiced_ratio": voiced_ratio,
            "pitch_stability": pitch_stability,
            "spectral_flatness_score": flatness_score,
            "zcr_score": zcr_score,
        },
    }


def segment_vocals_soft(
    vocals: np.ndarray,
    sr: int,
    padding_sec: float = 0.18,
    frame_length: int = 2048,
    hop_length: int = 256,
    smoothing_sec: float = 0.10,
    enter_threshold: float = 0.52,
    keep_threshold: float = 0.30,
    min_segment_sec: float = 0.45,
    max_gap_sec: float = 0.22,
    max_segment_sec: float = 8.0,
    long_valley_threshold: float = 0.30,
) -> list[dict]:
    """
    对 Demucs 分离后的人声做细粒度切片，并为每个片段生成软分数。

    参数:
    vocals: 人声波形，形状为 `(2, T)` 或 `(T,)`
    sr: 采样率
    padding_sec: 片段首尾补边时长，避免切得太紧
    frame_length: RMS 分析窗长度
    hop_length: 帧移
    smoothing_sec: activity 平滑窗口时长
    enter_threshold: 进入片段的高阈值
    keep_threshold: 维持片段的低阈值
    min_segment_sec: 最小片段时长
    max_gap_sec: 允许合并的短停顿时长
    max_segment_sec: 最大片段时长
    long_valley_threshold: 长片段内部二次切分的谷值阈值

    说明:
    整体流程是:
    1. 从人声 stem 中构造帧级 activity
    2. 用双阈值滞回找候选区间
    3. 合并过短停顿
    4. 对过长区间按谷值继续切细
    5. 对每个片段打软分数
    """
    if vocals.ndim == 1:
        vocals_mono = vocals.astype(np.float32)
        main_vocal = vocals_mono
        mid = vocals_mono
        side = np.zeros_like(vocals_mono)
    else:
        main_vocal, mid, side = extract_main_vocal(vocals)

    activity_profile = _build_activity_profile(
        main_vocal=main_vocal,
        mid=mid,
        side=side,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        smoothing_sec=smoothing_sec,
    )

    intervals = _segment_activity_frames(
        activity=activity_profile["activity"],
        split_activity=activity_profile["split_activity"],
        sr=sr,
        hop_length=hop_length,
        enter_threshold=enter_threshold,
        keep_threshold=keep_threshold,
        min_segment_sec=min_segment_sec,
        max_gap_sec=max_gap_sec,
        max_segment_sec=max_segment_sec,
        long_valley_threshold=long_valley_threshold,
    )

    segments = []
    for start_frame, end_frame in intervals:
        start_sample = int(start_frame * hop_length)
        end_sample = int(end_frame * hop_length)
        segment = _score_segment(
            main_vocal=main_vocal,
            mid=mid,
            side=side,
            sr=sr,
            start_sample=start_sample,
            end_sample=end_sample,
            activity_profile=activity_profile,
            hop_length=hop_length,
            padding_sec=padding_sec,
        )
        if segment["duration"] >= min_segment_sec:
            segments.append(segment)
    return segments


def analyze_vocal_file(
    music_path: str,
    padding_sec: float = 0.18,
    min_segment_sec: float = 0.45,
    max_gap_sec: float = 0.22,
    max_segment_sec: float = 8.0,
) -> dict:
    """
    对音频文件执行完整的人声分析流程。

    参数:
    music_path: 音频文件路径
    padding_sec: 片段首尾补边时长
    min_segment_sec: 最小片段时长
    max_gap_sec: 允许合并的短停顿时长
    max_segment_sec: 最大片段时长
    """
    separated = separate_vocals_file(music_path)
    segments = segment_vocals_soft(
        separated["vocals_stereo"],
        sr=separated["sr"],
        padding_sec=padding_sec,
        min_segment_sec=min_segment_sec,
        max_gap_sec=max_gap_sec,
        max_segment_sec=max_segment_sec,
    )
    is_vocal, vocal_time = has_vocals_from_array(
        separated["main_vocal"],
        sr=separated["sr"],
        origin=separated["reference_mono"],
    )
    return {
        "music_path": music_path,
        "sr": separated["sr"],
        "is_vocal": is_vocal,
        "vocal_time": vocal_time,
        "main_vocal": separated["main_vocal"],
        "vocals_stereo": separated["vocals_stereo"],
        "segments": segments,
    }
