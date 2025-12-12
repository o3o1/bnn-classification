import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm
import shutil
import random
import os

TARGET_SR = 4000
SEGMENT_SIZE = 4096   # 1秒
INPUT_ROOT = Path("dataset")
OUT_DIR_A = Path("dataset_global_v16")
OUT_DIR_B = Path("dataset_instance_v6")
LABEL_MAP = {"通畅": 0, "轻度阻塞": 1, "重度阻塞": 2}

def find_global_max_dedc(input_root):
    print("扫描全局最大值...")
    max_val = 0.0
    wav_files = list(input_root.glob("**/*.wav"))
    sampled = random.sample(wav_files, max(1, int(len(wav_files)*0.3)))
    for f in tqdm(sampled):
        try:
            sr, s = wavfile.read(f)
            s = s.astype(np.float32)
            if np.abs(s).max()>1.0: s/=32768.0
            s = s - np.mean(s)
            max_val = max(max_val, np.max(np.abs(s)))
        except: pass
    if max_val == 0: max_val = 1.0
    print(f"Global Max: {max_val}")
    return max_val

def load_audio(path):
    try:
        sr, samples = wavfile.read(path)
        samples = samples.astype(np.float32)
        if np.abs(samples).max() > 1.0: samples /= 32768.0
        samples = samples - np.mean(samples)
        if sr != TARGET_SR:
            import math
            gcd = math.gcd(sr, TARGET_SR)
            samples = resample_poly(samples, TARGET_SR // gcd, sr // gcd)
        return samples
    except: return None

def instance_norm(x):
    std = np.std(x)
    if std < 1e-8: return np.zeros_like(x)
    return (x - np.mean(x)) / std

def process_dataset():
    if OUT_DIR_A.exists(): shutil.rmtree(OUT_DIR_A)
    OUT_DIR_A.mkdir(parents=True)
    if OUT_DIR_B.exists(): shutil.rmtree(OUT_DIR_B)
    OUT_DIR_B.mkdir(parents=True)
    
    global_max = find_global_max_dedc(INPUT_ROOT)
    
    for split in ["train", "val", "test"]:
        print(f"处理 {split} ...")
        ax, ay = [], []
        bx, by = [], []
        
        for class_name, label_idx in LABEL_MAP.items():
            search_path = INPUT_ROOT / split / class_name
            wav_files = list(search_path.glob("*.wav"))
            for wav_path in tqdm(wav_files, desc=class_name):
                raw = load_audio(wav_path)
                if raw is None or len(raw) < SEGMENT_SIZE: continue
                
                step = SEGMENT_SIZE // 2 
                for i in range(0, len(raw)-SEGMENT_SIZE+1, step):
                    window = raw[i : i+SEGMENT_SIZE]
                    
                    norm_a = np.clip(window / global_max, -1.0, 1.0)
                    ax.append(norm_a)
                    ay.append(label_idx)
                    
                    if label_idx in [1, 2]:
                        norm_b = instance_norm(window)
                        target_b = 0 if label_idx == 1 else 1
                        bx.append(norm_b)
                        by.append(target_b)

        if len(ax)>0:
            np.savez_compressed(OUT_DIR_A / f"{split}.npz", x=np.array(ax, dtype=np.float32), y=np.array(ay, dtype=np.int64))
        if len(bx)>0:
            np.savez_compressed(OUT_DIR_B / f"{split}.npz", x=np.array(bx, dtype=np.float32), y=np.array(by, dtype=np.int64))

if __name__ == "__main__":
    process_dataset()