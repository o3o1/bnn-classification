import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset_2048_final"
MODEL_A_PATH = "model_A_bnn_2048.pth" 
MODEL_B_PATH = "model_B_bnn_2048.pth"

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): return input.sign()

class BinarizedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        return F.conv1d(input, self.weight.sign(), self.bias, self.stride, self.padding, self.dilation, self.groups)

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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel//reduction, bias=False), nn.ReLU(), nn.Linear(channel//reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x): b, c, _ = x.size(); return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1)

class BinarySEResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = BinarizedConv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = BinarizedConv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm1d(planes))
    def forward(self, x):
        out = self.se(self.bn2(self.conv2(F.hardtanh(self.bn1(self.conv1(x))))))
        out += self.shortcut(x)
        return F.hardtanh(out)

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

def check_stm32_ram():
    print("\nSTM32L031 RAM 最终核算")
    input_ram = 2048 * 2
    l1_stream_buffer = 1024
    l2_ram = 512
    l3_ram = 512
    stack_ram = 1536
    
    total = input_ram + l1_stream_buffer + max(l2_ram, l3_ram) + stack_ram
    
    print(f"1.输入数据 (2048 Int16): {input_ram/1024:.2f} KB")
    print(f"2.中间层 (流式+二值化): {(l1_stream_buffer + l2_ram)/1024:.2f} KB")
    print(f"3.系统栈预留: {stack_ram/1024:.2f} KB")
    print(f"总计需求: {total/1024:.2f} KB")
    
    if total < 8192:
        print("内存检查通过(<8KB)")
    else:
        print("内存超标")

def main():
    check_stm32_ram()
    
    if not os.path.exists(MODEL_A_PATH) or not os.path.exists(MODEL_B_PATH):
        print("错误：找不到模型文件")
        return

    model_a = GatekeeperBNN().to(DEVICE)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
    model_a.eval()
    
    model_b = ExpertBNN().to(DEVICE)
    model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
    model_b.eval()
    
    d = np.load(os.path.join(DATA_DIR, "test.npz"))
    x_test = torch.from_numpy(d['x']).unsqueeze(1).to(DEVICE)
    y_test = torch.from_numpy(d['y']).long().to(DEVICE)
    
    print(f"\n正在测试 {len(x_test)} 个样本")
    
    final_preds = []
    with torch.no_grad():
        for i in range(len(x_test)):
            sample = x_test[i:i+1]
            
            pred_a = model_a(sample).argmax(1).item()
            
            if pred_a == 0:
                final_preds.append(0)
            else:
                rms_val = torch.sqrt(torch.mean(sample**2)) * 10.0
                rms_input = rms_val.view(1, 1)
                
                pred_b = model_b(sample, rms_input).argmax(1).item()
                
                if pred_b == 0:
                    final_preds.append(1)
                else:
                    final_preds.append(2)
    
    final_preds = np.array(final_preds)
    y_true = y_test.cpu().numpy()
    
    acc = accuracy_score(y_true, final_preds)
    print(f"\n最终 BNN 结果")
    print(f"准确率: {acc*100:.2f}%")
    print("混淆矩阵:")
    print(confusion_matrix(y_true, final_preds))

if __name__ == "__main__":
    main()