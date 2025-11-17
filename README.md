# BNN二分类

流式数据预处理脚本 `preprocess/streaming_segments.py`。它会遍历指定目录下的 4 kHz 呼吸音频片段，完成如下步骤：

1. 统一降采样到 2 kHz 并按 4 秒划分为 **clip**（例如 1 分钟录音可形成 15 个 clip）。
2. 每个 clip 再切分为长度 128 点的 **segment**，同时在 segment 前拼接感受野所需的历史数据（`R-1` 个点，自动依据 BNN 结构计算，**记得将数据替换为自己的BNN结构**）。
3. 将每个拼接后的窗口独立保存为 `NPZ` 文件，文件名格式为 `原文件名-clipXXX-segYYY.npz`，其中 `clip` 表示 4 秒片段编号，`seg` 表示该片段内的第几个 segment。
4. NPZ 内容包含窗口数据 `window`、clip/segment 元信息、起始时间戳、采样率、`STREAM_SEGMENT` 与 `ReceptiveField` 等字段。

## 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（推荐使用 `uv` 管理环境）

## 安装环境

```bash
uv sync
```

或者使用`pip`

```bash
pip install -r requirements.txt
```

## 命令行使用

```bash
# 处理 example/ 目录下的 wav，结果保存到 stream_samples/
uv run python -m preprocess.streaming_segments example --output-dir stream_samples --verbose
```

- `example`：包含一系列 4 kHz的 wav 文件。
- `--output-dir`：NPZ 输出目录，默认为 `./stream_samples`。
- `--verbose`：开启后打印每个 clip/segment 的处理日志。

运行结束后，可在输出目录中看到类似 `xxx-clip001-seg003.npz` 的文件。

## 文件内调用

仓库根目录的 `main.py` 提供了最简示例，会调用 `process_directory` 处理数据，并读取首个 NPZ 文件，打印窗口数据及元信息。

```python
from pathlib import Path
from preprocess.streaming_segments import process_directory

INPUT_DIR = Path("example")                                      # 输入目录（4 kHz的 wav 片段）
OUTPUT_DIR = Path("output")                                      # 输出目录（保存为npz）
VERBOSE = True                                                   # 是否打印过程日志

saved_paths = process_directory(INPUT_DIR, OUTPUT_DIR, VERBOSE)  # 包含输出文件路径的list


```

## 读取样例

```python
import numpy as np
from pathlib import Path

path = Path("stream_samples/foo-clip001-seg001.npz")
data = np.load(path)
window = data["window"]          # ndarray，长度 = receptive_field - 1 + 128
clip_idx = data["clip_index"]    # clip 序号（4 秒片段）
seg_idx = data["segment_index"]  # segment 序号（clip 内的第几个 128 点窗口）
```
