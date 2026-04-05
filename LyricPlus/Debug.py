import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from LyricPlus.Align import LyricAlignmentResult
from LyricPlus.Lyric import LyricLineStamp
from LyricPlus.Transcribe import WhisperTrackResult


def format_time(seconds: float) -> str:
    """
    将秒数格式化为 `MM:SS.xx`。

    参数:
        seconds: 秒数。
    """
    minute = int(seconds // 60)
    second = seconds % 60
    return f"{minute:02d}:{second:05.2f}"


def format_time_filename(seconds: float) -> str:
    """
    将秒数格式化为适合文件名的字符串。

    参数:
        seconds: 秒数。
    """
    return format_time(seconds).replace(":", "-").replace(".", "_")


def _read_embedded_mp4_lyric(music_path: str | None) -> str | None:
    """
    从 m4a 的 `©lyr` 标签读取内嵌歌词。

    参数:
        music_path: 音频文件路径。
    """
    if not music_path:
        return None

    path = Path(music_path)
    if path.suffix.lower() != ".m4a" or not path.exists():
        return None

    from mutagen.mp4 import MP4

    tags = MP4(str(path))
    values = tags.tags.get("\xa9lyr") if tags.tags else None
    if not values:
        return None
    if isinstance(values, list):
        values = [str(item).strip() for item in values if str(item).strip()]
        return "\n".join(values).strip() or None
    text = str(values).strip()
    return text or None


def load_lyric_hint(lyric_path: str | None, music_path: str | None = None):
    """
    读取可选歌词提示。

    参数:
        lyric_path: 歌词文件路径，可为 `.lrc` 或纯文本。
        music_path: 未传入外部歌词时，用于读取 m4a 内嵌歌词。
    """
    if not lyric_path:
        embedded = _read_embedded_mp4_lyric(music_path)
        if embedded is None:
            return None
        if re.search(r"\[\d{2,}:\d{2}\.\d{2,3}\]", embedded):
            return LyricLineStamp(embedded)
        return embedded

    path = Path(lyric_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到歌词提示文件: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".lrc":
        return LyricLineStamp(text)
    return text


def load_alignment_lyric(lyric_path: str | None, music_path: str | None = None) -> LyricLineStamp | None:
    """
    读取用于约束对齐的歌词对象。

    参数:
        lyric_path: 外部歌词路径，可为空。
        music_path: 未传入外部歌词时，用于读取 m4a 内嵌歌词。
    """
    lyric_data = load_lyric_hint(lyric_path, music_path=music_path)
    if lyric_data is None:
        return None
    if isinstance(lyric_data, LyricLineStamp):
        return lyric_data
    if re.search(r"\[\d{2,}:\d{2}\.\d{2,3}\]", lyric_data):
        return LyricLineStamp(lyric_data)
    return LyricLineStamp.from_plain_text(lyric_data)


def make_json_safe(value):
    """
    将包含 numpy 标量等对象递归转换为 JSON 可序列化结构。

    参数:
        value: 任意待序列化对象。
    """
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_vocal_profile(profile: dict, segments: list[dict], initial_intervals: list[tuple[int, int]]):
    """
    打印活动曲线统计和分片摘要。

    参数:
        profile: `VocalAnalyzer.build_activity_profile` 返回的特征字典。
        segments: 分片结果字典列表。
        initial_intervals: 初次滞回分片区间。
    """
    activity = profile["activity"]
    split_activity = profile["split_activity"]
    print(f"frames: {len(activity)}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"activity_p{p}: {float(np.percentile(activity, p)):.6f}")
    print(
        "center_long_p5/p50/p95:",
        f"{float(np.percentile(profile['center_score_long'], 5)):.6f}",
        f"{float(np.percentile(profile['center_score_long'], 50)):.6f}",
        f"{float(np.percentile(profile['center_score_long'], 95)):.6f}",
    )
    print(
        "split_activity_p5/p50/p95:",
        f"{float(np.percentile(split_activity, 5)):.6f}",
        f"{float(np.percentile(split_activity, 50)):.6f}",
        f"{float(np.percentile(split_activity, 95)):.6f}",
    )
    print(f"initial_intervals: {len(initial_intervals)}")
    print(f"final_segments: {len(segments)}")
    print("segments:\n")

    for idx, seg in enumerate(segments, start=1):
        scores = seg["scores"]
        features = seg["features"]
        print(
            {
                "idx": idx,
                "start": format_time(seg["start"]),
                "end": format_time(seg["end"]),
                "duration": round(seg["duration"], 2),
                "core_start": format_time(seg["core_start"]),
                "core_end": format_time(seg["core_end"]),
                "core_duration": round(seg["core_duration"], 2),
                "presence": round(scores["vocal_presence"], 3),
                "off_center_risk": round(scores["off_center_risk"], 3),
                "center_long_q80": round(features["center_long_q80"], 3),
                "center_low_ratio": round(features["center_low_ratio"], 3),
            }
        )


def plot_vocal_profile(
    profile: dict,
    initial_intervals: list[tuple[int, int]],
    segments: list[dict],
    sr: int,
    hop_length: int,
    output_path: Path,
    title: str,
    keep_threshold: float,
    enter_threshold: float,
):
    """
    绘制 activity 曲线、分片结果和分片级分数。
    """
    t = profile["frame_times"]
    step = max(len(t) // 5000, 1)

    fig, axes = plt.subplots(7, 1, figsize=(16, 15), sharex=True)
    axes[0].plot(t[::step], profile["energy_score"][::step], linewidth=0.8, color="tab:blue")
    axes[0].set_ylabel("energy")

    axes[1].plot(t[::step], profile["center_score"][::step], linewidth=0.5, color="tab:green", alpha=0.35)
    axes[1].plot(t[::step], profile["center_score_smooth"][::step], linewidth=1.0, color="tab:olive")
    axes[1].plot(t[::step], profile["center_score_long"][::step], linewidth=1.0, color="goldenrod")
    axes[1].set_ylabel("center")

    axes[2].plot(t[::step], profile["activity"][::step], linewidth=0.8, color="tab:red")
    axes[2].axhline(keep_threshold, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(enter_threshold, color="gray", linestyle=":", linewidth=1)
    axes[2].set_ylabel("activity")

    axes[3].plot(t[::step], profile["split_activity"][::step], linewidth=0.9, color="tab:purple")
    axes[3].set_ylabel("split")

    axes[4].plot(t[::step], profile["activity"][::step], linewidth=0.45, color="lightcoral", alpha=0.55)
    axes[4].plot(t[::step], profile["split_activity"][::step], linewidth=1.0, color="tab:red")
    for start_frame, end_frame in initial_intervals:
        start_t = start_frame * hop_length / sr
        end_t = end_frame * hop_length / sr
        axes[4].axvspan(start_t, end_t, color="tab:blue", alpha=0.08)
    for seg in segments:
        axes[4].axvline(seg["start"], color="tab:gray", alpha=0.15, linewidth=0.8)
    axes[4].set_ylabel("segments")

    centers = np.array([(seg["start"] + seg["end"]) / 2 for seg in segments], dtype=np.float32)
    presence_scores = np.array([seg["scores"]["vocal_presence"] for seg in segments], dtype=np.float32)
    off_center_scores = np.array([seg["scores"]["off_center_risk"] for seg in segments], dtype=np.float32)
    center_long_q80 = np.array([seg["features"]["center_long_q80"] for seg in segments], dtype=np.float32)
    center_low_ratio = np.array([seg["features"]["center_low_ratio"] for seg in segments], dtype=np.float32)

    axes[5].plot(centers, presence_scores, marker="o", markersize=3, linewidth=1.0, color="tab:blue", label="presence")
    axes[5].plot(centers, off_center_scores, marker="o", markersize=2, linewidth=0.9, color="tab:red", label="off_center")
    for idx, (x, y) in enumerate(zip(centers, presence_scores), start=1):
        axes[5].text(x, min(y + 0.03, 0.98), str(idx), fontsize=6, color="tab:blue", ha="center")
    axes[5].legend(loc="upper right", ncol=2, fontsize=8)
    axes[5].set_ylabel("scores")

    axes[6].plot(centers, center_long_q80, marker="o", markersize=2, linewidth=0.9, color="tab:brown", label="center_long_q80")
    axes[6].plot(centers, center_low_ratio, marker="o", markersize=2, linewidth=0.9, color="tab:green", label="center_low")
    for idx, x in enumerate(centers, start=1):
        axes[6].text(x, 0.98, str(idx), fontsize=6, color="tab:gray", ha="center", va="top")
    axes[6].legend(loc="upper right", ncol=2, fontsize=8)
    axes[6].set_ylabel("features")
    axes[6].set_xlabel("time (s)")

    axes[0].set_title(title)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_vocal_segments(result, output_dir: str | Path) -> Path:
    """
    将分片结果导出为 wav 文件。

    参数:
        result: `VocalAnalyzer.analyze_file()` 返回结果对象。
        output_dir: 根导出目录。
    """
    export_root = Path(output_dir) / Path(result.music_path).stem
    export_root.mkdir(parents=True, exist_ok=True)

    for idx, segment in enumerate(result.segments, start=1):
        filename = (
            f"{idx:03d}_"
            f"{format_time_filename(segment.start)}_"
            f"{format_time_filename(segment.end)}_"
            f"p{segment.scores['vocal_presence']:.2f}_"
            f"r{segment.scores['off_center_risk']:.2f}.wav"
        )
        out_path = export_root / filename
        audio = torch.from_numpy(segment.audio).unsqueeze(0)
        torchaudio.save(str(out_path), audio, result.sr)

    return export_root


def summarize_transcription(vocal_result: dict, track_result: WhisperTrackResult):
    """
    打印整首歌的分片转写摘要。

    参数:
        vocal_result: `VocalAnalyzer` 分析结果字典。
        track_result: `WhisperTranscriber` 聚合结果对象。
    """
    print(f"music: {vocal_result['music_path']}")
    print(f"is_vocal: {vocal_result['is_vocal']}")
    print(f"vocal_time: {vocal_result['vocal_time']:.2f}s")
    print(f"segments: {len(vocal_result['segments'])}")
    print(f"transcribed_chunks: {len(track_result.chunks)}")
    print(f"stats: {track_result.stats}")
    print("\nchunks:\n")

    for chunk in track_result.chunks:
        scores = chunk.scores
        print(
            {
                "idx": chunk.segment_index,
                "start": format_time(chunk.start),
                "end": format_time(chunk.end),
                "core_start": format_time(chunk.core_start),
                "core_end": format_time(chunk.core_end),
                "presence": round(scores.get("vocal_presence", 0.0), 3),
                "off_center_risk": round(scores.get("off_center_risk", 0.0), 3),
                "avg_confidence": None if chunk.avg_confidence is None else round(chunk.avg_confidence, 3),
                "avg_logprob": None if chunk.avg_logprob is None else round(chunk.avg_logprob, 3),
                "confidence_source": chunk.confidence_source,
                "min_token_confidence": None if chunk.min_token_confidence is None else round(chunk.min_token_confidence, 3),
                "low_conf_token_ratio": None if chunk.low_conf_token_ratio is None else round(chunk.low_conf_token_ratio, 3),
                "compression_ratio": round(chunk.compression_ratio, 3),
                "self_repeat_score": round(chunk.self_repeat_score, 3),
                "hallucination_risk": round(chunk.hallucination_risk, 3),
                "risk_components": {k: round(v, 3) for k, v in chunk.risk_components.items()},
                "token_count": 0 if not chunk.token_confidences else len(chunk.token_confidences),
                "num_candidates": chunk.raw_result.get("num_candidates", 1),
                "selected_index": chunk.raw_result.get("selected_index", 0),
                "prompt_text": chunk.prompt_text,
                "text": chunk.text,
            }
        )


def save_transcription_json(output_path: Path, vocal_result: dict, track_result: WhisperTrackResult):
    """
    将转写测试结果保存为 JSON。

    参数:
        output_path: 输出文件路径。
        vocal_result: `VocalAnalyzer` 分析结果字典。
        track_result: `WhisperTranscriber` 聚合结果对象。
    """
    transcription = track_result.to_dict()
    for chunk in transcription.get("chunks", []):
        chunk.pop("raw_result", None)

    payload = {
        "music_path": vocal_result["music_path"],
        "sr": vocal_result["sr"],
        "is_vocal": vocal_result["is_vocal"],
        "vocal_time": vocal_result["vocal_time"],
        "segment_count": len(vocal_result["segments"]),
        "transcription": transcription,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_transcription_profile(
    vocal_result: dict,
    track_result: WhisperTrackResult,
    output_path: Path,
    title: str,
):
    """
    绘制分片转写结果、总分和底层风险组件。
    """
    segments = vocal_result["segments"]
    chunks_by_index = {chunk.segment_index: chunk for chunk in track_result.chunks}
    used_segments = [seg for idx, seg in enumerate(segments, start=1) if idx in chunks_by_index]
    used_chunks = [chunks_by_index[idx] for idx in range(1, len(segments) + 1) if idx in chunks_by_index]

    if not used_segments:
        return

    t = vocal_result["activity_profile"]["frame_times"]
    activity = vocal_result["activity_profile"]["activity"]
    step = max(len(t) // 5000, 1)

    centers = np.array([(seg["start"] + seg["end"]) / 2 for seg in used_segments], dtype=np.float32)
    avg_confidence = np.array(
        [chunk.avg_confidence if chunk.avg_confidence is not None else 0.0 for chunk in used_chunks],
        dtype=np.float32,
    )
    hallucination_risk = np.array([chunk.hallucination_risk for chunk in used_chunks], dtype=np.float32)

    component_names = [
        "min_token_risk",
        "low_conf_ratio_risk",
        "off_center_risk",
        "presence_risk",
        "self_repeat_risk",
        "avg_confidence_risk",
        "compression_risk",
        "density_risk",
    ]
    component_colors = {
        "min_token_risk": "tab:red",
        "low_conf_ratio_risk": "tab:orange",
        "off_center_risk": "tab:purple",
        "presence_risk": "tab:blue",
        "self_repeat_risk": "tab:brown",
        "avg_confidence_risk": "tab:green",
        "compression_risk": "tab:pink",
        "density_risk": "tab:gray",
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(t[::step], activity[::step], linewidth=0.8, color="tab:red", alpha=0.9)
    for idx, seg in enumerate(used_segments, start=1):
        axes[0].axvspan(seg["start"], seg["end"], color="tab:blue", alpha=0.10)
        axes[0].text((seg["start"] + seg["end"]) / 2, 0.98, str(idx), fontsize=6, color="tab:gray", ha="center", va="top")
    axes[0].set_ylabel("segments")

    axes[1].plot(centers, avg_confidence, marker="o", markersize=3, linewidth=1.0, color="tab:blue", label="avg_confidence")
    axes[1].plot(centers, hallucination_risk, marker="o", markersize=3, linewidth=1.0, color="tab:red", label="hallucination_risk")
    for idx, (x, y) in enumerate(zip(centers, hallucination_risk), start=1):
        axes[1].text(x, min(y + 0.03, 0.98), str(idx), fontsize=6, color="tab:red", ha="center")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[1].set_ylabel("scores")

    for name in component_names:
        values = np.array([chunk.risk_components.get(name, 0.0) for chunk in used_chunks], dtype=np.float32)
        axes[2].plot(centers, values, marker="o", markersize=2, linewidth=0.9, color=component_colors[name], label=name)
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    axes[2].set_ylabel("risk comps")
    axes[2].set_xlabel("time (s)")

    axes[0].set_title(title)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_alignment(alignment_result: LyricAlignmentResult):
    """
    打印歌词约束对齐摘要。

    参数:
        alignment_result: `LyricAligner.align()` 返回结果。
    """
    print(f"music: {alignment_result.music_path}")
    print(f"stats: {alignment_result.stats}")
    print("\naligned lines:\n")

    for item in alignment_result.items:
        print(
            {
                "line_index": item.line_index,
                "time": None if item.start is None else f"{format_time(item.start)} -> {format_time(item.end)}",
                "chunk_indices": item.chunk_indices,
                "similarity": round(item.similarity, 3),
                "score": round(item.score, 3),
                "confidence": None if item.confidence is None else round(item.confidence, 3),
                "hallucination_risk": None if item.hallucination_risk is None else round(item.hallucination_risk, 3),
                "lyric": item.line.text,
                "chunk_text": item.chunk_text,
            }
        )


def save_alignment_json(output_path: Path, alignment_result: LyricAlignmentResult):
    """
    将歌词约束对齐结果保存为 JSON。

    参数:
        output_path: 输出文件路径。
        alignment_result: `LyricAligner.align()` 返回结果。
    """
    payload = alignment_result.to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
