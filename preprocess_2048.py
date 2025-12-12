import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm
import shutil
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_SR = 4000
SEGMENT_SIZE = 2048
INPUT_ROOT = Path("dataset")
OUTPUT_DIR = Path("dataset_2048_final")
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

def process_dataset():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    global_max = find_global_max_dedc(INPUT_ROOT)
    
    for split in ["train", "val", "test"]:
        print(f"处理 {split} ...")
        all_x, all_y = [], []
        
        for class_name, label_idx in LABEL_MAP.items():
            search_path = INPUT_ROOT / split / class_name
            wav_files = list(search_path.glob("*.wav"))
            for wav_path in tqdm(wav_files, desc=class_name):
                raw = load_audio(wav_path)
                if raw is None or len(raw) < SEGMENT_SIZE: continue
                
                step = SEGMENT_SIZE // 2 
                for i in range(0, len(raw)-SEGMENT_SIZE+1, step):
                    window = raw[i : i+SEGMENT_SIZE]
                    
                    norm = window / global_max
                    norm = np.clip(norm, -1.0, 1.0)
                    
                    all_x.append(norm)
                    all_y.append(label_idx)
        
        if len(all_x)>0:
            np.savez_compressed(
                OUTPUT_DIR / f"{split}.npz",
                x=np.array(all_x, dtype=np.float32),
                y=np.array(all_y, dtype=np.int64)
            )

if __name__ == "__main__":
    process_dataset()