import torch
import torch.nn as nn
import numpy as np
import os
import random
from sklearn.metrics import confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(1, 32, 64, 4, 30, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 17, 2, 8, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 9, 2, 4, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(nn.Conv1d(128, 128, 5, 2, 2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 3))
    def forward(self, x):
        return self.fc(self.gap(self.layer4(self.layer3(self.layer2(self.layer1(x))))).view(x.size(0), -1))

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(c, c//r, bias=False), nn.ReLU(), nn.Linear(c//r, c, bias=False), nn.Sigmoid())
    
    def forward(self, x):
        b, c, _ = x.size() 
        return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1)

class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(1,32,32,2,15,bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.l1 = self._b(32,64); self.l2 = self._b(64,128); self.l3 = self._b(128,128)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 2))
    def _b(self, i, o):
        return nn.Sequential(
            nn.Conv1d(i,o,3,2,1,bias=False), nn.BatchNorm1d(o), nn.ReLU(),
            nn.Conv1d(o,o,3,1,1,bias=False), nn.BatchNorm1d(o), SEBlock(o), nn.ReLU()
        )
    def forward(self, x): return self.fc(self.l3(self.l2(self.l1(self.c1(x)))))

def instance_norm_tensor(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-8
    return (x - mean) / std

SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为 {seed}")

def main():
    set_seed(SEED)
    print("最终融合推理")
    
    # 加载模型
    model_a = FeatureExtractorCNN().to(DEVICE)
    if os.path.exists("model_A.pth"):
        model_a.load_state_dict(torch.load("model_A.pth"))
    else:
        print("错误：找不到 model_A.pth")
        return
    model_a.eval()
    
    model_b = SEResNet().to(DEVICE)
    if os.path.exists("model_B.pth"):
        model_b.load_state_dict(torch.load("model_B.pth"))
    else:
        print("错误：找不到 model_B.pth")
        return
    model_b.eval()
    
    # 加载测试数据
    test_path = "dataset_global_v16/test.npz"
    if not os.path.exists(test_path):
        print(f"错误：找不到测试数据 {test_path}")
        return
        
    d = np.load(test_path)
    x_test = torch.from_numpy(d['x']).unsqueeze(1).to(DEVICE)
    y_test = torch.from_numpy(d['y']).long().to(DEVICE)
    
    final_preds = []
    
    print(f"开始推理 {len(x_test)} 个样本...")
    with torch.no_grad():
        for i in range(len(x_test)):
            sample = x_test[i:i+1]
            
            logits_a = model_a(sample)
            pred_a = torch.argmax(logits_a, 1).item()
            
            if pred_a == 0: 
                final_preds.append(0)
            else:
                sample_b = instance_norm_tensor(sample)
                logits_b = model_b(sample_b)
                pred_b = torch.argmax(logits_b, 1).item()
                
                if pred_b == 0:
                    final_preds.append(1)
                else:
                    final_preds.append(2)
    
    final_preds = np.array(final_preds)
    y_true = y_test.cpu().numpy()
    
    acc = np.mean(final_preds == y_true)
    print(f"\n最终成绩")
    print(f"Total Accuracy: {acc*100:.2f}%")
    print("混淆矩阵:\n", confusion_matrix(y_true, final_preds))

if __name__ == "__main__":
    main()