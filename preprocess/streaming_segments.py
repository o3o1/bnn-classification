import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


EXPECTED_SAMPLE_RATE = 4000
TARGET_SAMPLE_RATE = 2000
SAMPLE_DURATION = 4.0
SAMPLE_SIZE = int(TARGET_SAMPLE_RATE * SAMPLE_DURATION)
STREAM_SEGMENT = 128

# BNN 架构定义（用于计算感受野）
BNN_ARCHITECTURE = [
    {"kernel_size": 5, "stride": 1, "dilation": 1},
    {"kernel_size": 5, "stride": 1, "dilation": 2},
    {"kernel_size": 3, "stride": 2, "dilation": 1},
    {"kernel_size": 3, "stride": 1, "dilation": 1},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "遍历输入目录下的 4 kHz 呼吸片段，降采样到 2 kHz 后按 4 秒切片，"
            "并生成包含感受野历史的 128 点流式窗口，批量保存为 NPZ。"
        )
    )
    parser.add_argument("input_dir", type=Path, help="包含 4 kHz wav 片段的目录。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stream_samples"),
        help="输出目录，用于保存 NPZ 样本（默认 ./stream_samples）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出；默认静默运行。",
    )
    return parser.parse_args()


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sr, samples = wavfile.read(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    max_abs = np.max(np.abs(samples))
    if max_abs > 0:
        samples /= max_abs
    samples = (samples + 1.0) * 1.5
    return sr, samples


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    return resample_poly(samples, target_sr, orig_sr)


def compute_receptive_field(layers: Iterable[dict]) -> int:
    receptive_field = 1
    jump = 1
    for layer in layers:
        kernel = layer["kernel_size"]
        stride = layer.get("stride", 1)
        dilation = layer.get("dilation", 1)
        receptive_field += (kernel - 1) * jump * dilation
        jump *= stride
    return receptive_field


def iterate_samples(samples: np.ndarray) -> List[tuple[int, float, np.ndarray]]:
    total = len(samples) // SAMPLE_SIZE
    result: List[tuple[int, float, np.ndarray]] = []
    for idx in range(total):
        start = idx * SAMPLE_SIZE
        segment = samples[start : start + SAMPLE_SIZE]
        result.append((idx, start / TARGET_SAMPLE_RATE, segment))
    return result


def build_stream_windows(sample: np.ndarray, receptive_field: int) -> np.ndarray:
    history_len = max(receptive_field - 1, 0)
    windows: List[np.ndarray] = []
    total = len(sample) // STREAM_SEGMENT

    for seg_idx in range(total):
        seg_start = seg_idx * STREAM_SEGMENT
        seg_end = seg_start + STREAM_SEGMENT
        current = sample[seg_start:seg_end]

        hist_start = max(0, seg_start - history_len)
        history = sample[hist_start:seg_start]
        if history_len > 0 and len(history) < history_len:
            pad = np.zeros(history_len - len(history), dtype=sample.dtype)
            history = np.concatenate([pad, history])

        window = np.concatenate([history, current])
        windows.append(window)

    if not windows:
        return np.empty((0, history_len + STREAM_SEGMENT), dtype=sample.dtype)
    return np.stack(windows, axis=0)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stream_window(
    out_dir: Path,
    base_name: str,
    clip_idx: int,
    seg_idx: int,
    clip_start_time: float,
    segment_start_time: float,
    window: np.ndarray,
    receptive_field: int,
) -> Path:
    target = out_dir / f"{base_name}-clip{clip_idx:03d}-seg{seg_idx:03d}.npz"
    np.savez(
        target,
        window=window,
        clip_index=clip_idx,
        segment_index=seg_idx,
        clip_start_time=clip_start_time,
        segment_start_time=segment_start_time,
        sample_rate=TARGET_SAMPLE_RATE,
        stream_segment=STREAM_SEGMENT,
        receptive_field=receptive_field,
    )
    return target


def collect_wav_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"输入路径 {input_dir} 不是有效目录。")
    wav_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not wav_files:
        raise ValueError(f"目录 {input_dir} 中未找到 wav 文件。")
    return wav_files


def process_directory(input_dir: Path, output_dir: Path, verbose: bool = False) -> List[Path]:
    input_dir = input_dir.resolve()
    wav_files = collect_wav_files(input_dir)

    receptive_field = compute_receptive_field(BNN_ARCHITECTURE)
    history_len = max(receptive_field - 1, 0)

    out_dir = ensure_output_dir(output_dir.resolve())
    saved_paths: List[Path] = []

    if verbose:
        print(
            f"共找到 {len(wav_files)} 个文件。BNN 感受野 R = {receptive_field}，"
            f"窗口长度 = {STREAM_SEGMENT} + {history_len}(历史)。"
        )

    for wav_path in wav_files:
        sr, samples = load_audio(wav_path)
        if sr != EXPECTED_SAMPLE_RATE:
            raise ValueError(f"{wav_path.name} 采样率为 {sr} Hz，期望 {EXPECTED_SAMPLE_RATE} Hz。")

        samples_down = resample_audio(samples, sr, TARGET_SAMPLE_RATE)
        splitted = iterate_samples(samples_down)
        if not splitted:
            if verbose:
                print(f"警告：{wav_path.name} 长度不足 4 秒，跳过。")
            continue

        base_name = wav_path.stem
        for clip_idx, (_, clip_start_time, sample) in enumerate(splitted, start=1):
            if verbose:
                print(f"{wav_path.stem} 文件内 clip{clip_idx:03d} 起始 {clip_start_time:.2f}s。")

            windows = build_stream_windows(sample, receptive_field)
            if not len(windows):
                continue
            for seg_idx, window in enumerate(windows, start=1):
                segment_offset = (seg_idx - 1) * STREAM_SEGMENT / TARGET_SAMPLE_RATE
                segment_start_time = clip_start_time + segment_offset
                target = save_stream_window(
                    out_dir,
                    base_name,
                    clip_idx,
                    seg_idx,
                    clip_start_time,
                    segment_start_time,
                    window,
                    receptive_field,
                )
                saved_paths.append(target)
                if verbose:
                    print(
                        f"  {base_name} clip{clip_idx:03d} seg{seg_idx:03d} "
                        f"(segment start {segment_start_time:.3f}s) → 保存 {target.name}"
                    )

    return saved_paths


def main() -> None:
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, args.verbose)


if __name__ == "__main__":
    main()
