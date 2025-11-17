from pathlib import Path

import numpy as np

from preprocess.streaming_segments import process_directory


INPUT_DIR = Path("example")        # 输入目录（4 kHz的 wav 片段）
OUTPUT_DIR = Path("output")        # 输出目录（保存为npz）
VERBOSE = True                     # 是否打印过程日志


def verify_saved_file(path: Path) -> None:
    """读取保存的 npz 文件并打印具体数据内容。"""
    data = np.load(path)
    window = data["window"]
    clip_idx = int(data["clip_index"])
    seg_idx = int(data["segment_index"])
    clip_start = float(data["clip_start_time"])
    seg_start = float(data["segment_start_time"])
    sr = int(data["sample_rate"])
    stream_segment = int(data["stream_segment"])
    receptive_field = int(data["receptive_field"])

    print(f"验证 {path.name}:")
    print(f"  clip = {clip_idx}, seg = {seg_idx}")
    print(f"  clip 起始 = {clip_start:.2f}s, segment 起始 = {seg_start:.3f}s")
    print(f"  采样率 = {sr} Hz, STREAM_SEGMENT = {stream_segment}, Receptive Field = {receptive_field}")
    print(f"  实际窗口数据 (长度 {len(window)}):")
    print(window)


def main() -> None:
    saved_paths = process_directory(INPUT_DIR, OUTPUT_DIR, VERBOSE)
    if not saved_paths:
        print("未生成任何 NPZ 文件，请检查 INPUT_DIR 是否包含有效音频。")
        return

    verify_saved_file(saved_paths[0])


if __name__ == "__main__":
    main()
