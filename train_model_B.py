import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random

DATA_DIR = "dataset_instance_v6"
BATCH_SIZE = 64
EPOCHS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    print("训练模型 B ...")
    if not os.path.exists(os.path.join(DATA_DIR, "train.npz")): 
        print("错误：找不到数据，请先运行 preprocess_all.py")
        return
        
    d = np.load(os.path.join(DATA_DIR, "train.npz")); train_dl = DataLoader(TensorDataset(torch.from_numpy(d['x']).unsqueeze(1), torch.from_numpy(d['y']).long()), batch_size=BATCH_SIZE, shuffle=True)
    d = np.load(os.path.join(DATA_DIR, "test.npz")); test_dl = DataLoader(TensorDataset(torch.from_numpy(d['x']).unsqueeze(1), torch.from_numpy(d['y']).long()), batch_size=BATCH_SIZE)
    
    model = SEResNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.24, 1.0]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_acc = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = torch.roll(x, shifts=random.randint(0, 200), dims=2) + 0.05 * torch.randn_like(x)
            
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
        if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), "model_B.pth")
    print(f"Model B 完成，最佳 Acc: {best_acc*100:.2f}%")

if __name__ == "__main__": main()