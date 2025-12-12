import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random

DATA_DIR = "dataset_global_v16"
BATCH_SIZE = 64
EPOCHS = 40 
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
    print("训练模型 A ...")
    if not os.path.exists(os.path.join(DATA_DIR, "train.npz")):
        print("错误：找不到数据，请先运行 preprocess_all.py")
        return
       
    d = np.load(os.path.join(DATA_DIR, "train.npz")); train_dl = DataLoader(TensorDataset(torch.from_numpy(d['x']).unsqueeze(1), torch.from_numpy(d['y']).long()), batch_size=BATCH_SIZE, shuffle=True)
    d = np.load(os.path.join(DATA_DIR, "test.npz")); test_dl = DataLoader(TensorDataset(torch.from_numpy(d['x']).unsqueeze(1), torch.from_numpy(d['y']).long()), batch_size=BATCH_SIZE)
    
    model = FeatureExtractorCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.0]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            gain = 0.2 + 4.8 * torch.rand(x.size(0), 1, 1, device=DEVICE)
            x = torch.clamp(x*gain, -1, 1) + 0.005 * torch.randn_like(x)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                corr += (model(x).argmax(1) == y).sum().item(); tot += y.size(0)
        acc = corr/tot
        print(f"Ep {ep+1} | Test Acc: {acc*100:.2f}%")
        if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), "model_A.pth")
    print(f"Model A 完成，最佳 Acc: {best_acc*100:.2f}%")

if __name__ == "__main__": main()