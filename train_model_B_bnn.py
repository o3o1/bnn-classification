import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import random
import torch.nn.functional as F

SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
set_seed(SEED)

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): return input.sign()
    @staticmethod
    def backward(ctx, grad_output): return grad_output

class BinarizedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        if self.training:
            with torch.no_grad(): s = torch.mean(torch.abs(self.weight))
            w = Binarize.apply(self.weight) * s
        else: w = Binarize.apply(self.weight)
        return F.conv1d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ExpertBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 17, 2, 8, bias=False), nn.BatchNorm1d(16), nn.Hardtanh(), nn.MaxPool1d(2),
            BinarizedConv1d(16, 32, 9, 2, 4, bias=False), nn.BatchNorm1d(32), nn.Hardtanh(), nn.MaxPool1d(2),
            BinarizedConv1d(32, 64, 5, 2, 2, bias=False), nn.BatchNorm1d(64), nn.Hardtanh(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(65, 32),
            nn.Hardtanh(),
            nn.Linear(32, 2)
        )

    def forward(self, x, rms):
        feat = self.cnn(x).view(x.size(0), -1)
        combined = torch.cat([feat, rms], dim=1)
        return self.fc(combined)

class BinaryDataset(Dataset):
    def __init__(self, split):
        d = np.load(f"dataset_2048_final/{split}.npz")
        indices = np.where((d['y'] == 1) | (d['y'] == 2))[0]
        self.x = d['x'][indices].astype(np.float32)
        self.y = d['y'][indices].astype(np.int64)
        self.y = np.where(self.y == 1, 0, 1)
        
    def __len__(self): return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        rms = np.sqrt(np.mean(x**2)) * 10.0
        
        return (torch.from_numpy(x).unsqueeze(0), torch.tensor([rms], dtype=torch.float32)), torch.tensor(y, dtype=torch.long)

def main():
    print("训练 BNN-B ...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator()
    g.manual_seed(SEED)
    train_ds = BinaryDataset("train")
    test_ds = BinaryDataset("test")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
    test_dl = DataLoader(test_ds, batch_size=64)
    
    model = ExpertBNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    for ep in range(60):
        model.train()
        for (x, rms), y in train_dl:
            x, rms, y = x.to(DEVICE), rms.to(DEVICE), y.to(DEVICE)
            rand_gain = torch.rand(x.size(0), 1, 1, generator=g).to(DEVICE)
            x = x * (0.8 + 0.4 * rand_gain) 
            
            optimizer.zero_grad()
            output = model(x, rms)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for (x, rms), y in test_dl:
                x, rms, y = x.to(DEVICE), rms.to(DEVICE), y.to(DEVICE)
                corr += (model(x, rms).argmax(1) == y).sum().item(); tot += y.size(0)
        acc = corr/tot
        print(f"Ep {ep+1} | Acc: {acc*100:.2f}%")
        if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), "model_B_bnn_2048.pth")

if __name__ == "__main__": main()