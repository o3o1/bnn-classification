import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class BinarizedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        if self.training:
            with torch.no_grad(): s = torch.mean(torch.abs(self.weight))
            w = Binarize.apply(self.weight) * s
        else: w = Binarize.apply(self.weight)
        return F.conv1d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class GatekeeperBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv1d(1, 16, 32, 4, 14), nn.BatchNorm1d(16), nn.Hardtanh(), nn.MaxPool1d(2))
        self.l2 = nn.Sequential(BinarizedConv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.Hardtanh(), nn.MaxPool1d(2))
        self.l3 = nn.Sequential(BinarizedConv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.Hardtanh(), nn.MaxPool1d(2))
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

CONFIG = {"data_dir": "dataset_2048_final", "batch": 64, "epochs": 40}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(split):
    d = np.load(f"{CONFIG['data_dir']}/{split}.npz")
    x = torch.from_numpy(d['x']).unsqueeze(1)
    y = torch.from_numpy(d['y']).long()
    y_bin = torch.where(y == 0, 0, 1)
    return TensorDataset(x, y_bin)

def main():
    print("训练 BNN-A ...")
    g = torch.Generator()
    g.manual_seed(SEED)
    train_dl = DataLoader(get_data("train"), batch_size=CONFIG['batch'], shuffle=True, generator=g)
    test_dl = DataLoader(get_data("test"), batch_size=CONFIG['batch'])
    
    model = GatekeeperBNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.00042)
    
    best_acc = 0
    for ep in range(CONFIG['epochs']):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            rand_gain = torch.rand(x.size(0), 1, 1, generator=g).to(DEVICE)
            x = x * (0.5 + 2.0 * rand_gain)
            
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
        print(f"Ep {ep+1} | Acc: {acc*100:.2f}%")
        if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), "model_A_bnn_2048.pth")

if __name__ == "__main__": main()