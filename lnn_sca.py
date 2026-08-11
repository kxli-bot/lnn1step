import os
import h5py
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from tqdm import tqdm

# ==================== 设备设置 ====================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ==================== 读取数据 ====================
print("正在读取数据...")
with h5py.File('ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r') as f:
#with h5py.File('ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5', 'r') as f:
    x_train = f['Profiling_traces/traces'][:].astype(np.float32)
    y_train = f['Profiling_traces/labels'][:].astype(np.int64)
    x_test  = f['Attack_traces/traces'][:].astype(np.float32)
    y_test  = f['Attack_traces/labels'][:].astype(np.int64)

print(f"训练集：{x_train.shape}  测试集：{x_test.shape}")

# ==================== 预处理 ====================
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

x_train = x_train.reshape(-1, 700, 1)
x_test  = x_test.reshape(-1, 700, 1)

x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train)
x_test_t  = torch.tensor(x_test)
y_test_t  = torch.tensor(y_test)

train_ds = TensorDataset(x_train_t, y_train_t)
test_ds  = TensorDataset(x_test_t,  y_test_t)
train_loader = DataLoader(train_ds, batch_size=200, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=200)

# ==================== 搭建 LNN 模型 ====================
class LNN_SCA(nn.Module):
    def __init__(self):
        super(LNN_SCA, self).__init__()
        # 输入压缩层：700 → 64
        self.input_proj = nn.Sequential(
            nn.Linear(700, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # CfC 液态神经网络核心层
        wiring = AutoNCP(64, 32)  # 32个神经元，输出64维
        self.cfc = CfC(64, wiring, batch_first=True)
        # 输出分类层
        self.classifier = nn.Sequential(
            nn.Linear(32, 256)
        )

    def forward(self, x):
        # x: (batch, 700, 1)
        x = x.squeeze(-1)              # (batch, 700)
        x = self.input_proj(x)         # (batch, 64)
        x = x.unsqueeze(1)             # (batch, 1, 64)
        out, _ = self.cfc(x)           # (batch, 1, 64)
        out = out[:, -1, :]            # (batch, 64)
        out = self.classifier(out)     # (batch, 256)
        return out
#conv+cfc
# class LNN_SCA(nn.Module):
#     def __init__(self):
#         super(LNN_SCA, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=11, padding=5),
#             nn.ReLU(),
#         )
#         wiring = AutoNCP(64, 32)
#         self.cfc = CfC(64, wiring, batch_first=True)
#         self.classifier = nn.Sequential(nn.Linear(32, 256))

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = x.permute(0, 2, 1)
#         out, _ = self.cfc(x)
#         out = out[:, -1, :]
#         out = self.classifier(out)
#         return out
# class LNN_SCA(nn.Module):
#     def __init__(self):
#         super().__init__()
#         wiring = AutoNCP(64, 32)
#         self.cfc = CfC(1, wiring, batch_first=True)
#         self.classifier = nn.Linear(32, 256)

#     def forward(self, x):
#         # x shape: (batch, 700, 1)
#         out, _ = self.cfc(x)
#         return self.classifier(out[:, -1, :])   

# model = LNN_SCA().to(device)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"LNN 参数量：{total_params:,}")
# print(f"CNN 参数量：12,053,504")
# print(f"参数减少：{(1 - total_params/12053504)*100:.1f}%")

# ==================== 训练 ====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_acc_list = []
test_acc_list  = []

print("\n开始训练...")
for epoch in range(100):
    model.train()
    correct = total = 0
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100", ncols=80)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += len(y_batch)
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    scheduler.step()
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)
    test_acc = correct / total

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"Epoch {epoch+1}/100 | loss: {avg_loss:.4f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")
# ==================== 保存模型 ====================
# torch.save(model.state_dict(), 'lnn_model/lnn_baseline.pth')
# print("\n模型已保存到 lnn_model/lnn_baseline.pth")

# torch.save(model.state_dict(), 'lnn_model/lnn_desync50.pth')
# print("\n模型已保存到 lnn_model/lnn_desync50.pth")

# torch.save(model.state_dict(), 'lnn_model/lnn_conv_cfc_desync50.pth')
# print("\n模型已保存到 lnn_model/lnn_conv_cfc_desync50.pth")

# torch.save(model.state_dict(), 'lnn_model/lnn_700steps.pth')
# print("\n模型已保存到 lnn_model/lnn_700steps.pth")
# ==================== 画训练曲线 ====================
plt.figure(figsize=(8, 4))
plt.plot(train_acc_list, label='训练准确率')
plt.plot(test_acc_list,  label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('LNN 训练曲线')
plt.legend()
plt.tight_layout()
# plt.savefig('lnn_model/lnn_training_stand_curve.png')
# print("训练曲线已保存到 lnn_training_stand_curve.png")

# plt.title('LNN 训练曲线（lnn_conv_cfc_desync50）')
# plt.savefig('lnn_model/lnn_conv_cfc_desync50.png')

# plt.title('LNN 训练曲线（lnn_700_cfc）')
# plt.savefig('lnn_model/lnn_700_cfc.png')


print(f"LNN 参数量：{total_params:,}")
print(f"LNN 最终训练准确率：{train_acc_list[-1]:.4f}")
print(f"LNN 最终测试准确率：{test_acc_list[-1]:.4f}")