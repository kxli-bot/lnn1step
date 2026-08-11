import h5py
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ==================== 设备设置 ====================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ==================== 读取数据 ====================
#DATA_PATH = 'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5'
DATA_PATH = 'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5'
with h5py.File(DATA_PATH, 'r') as f:
    x_train  = np.array(f['Profiling_traces/traces'], dtype=np.int8).astype(np.float32)
    x_test   = np.array(f['Attack_traces/traces'],    dtype=np.int8).astype(np.float32)
    metadata = f['Attack_traces/metadata'][:]

correct_key = metadata['key'][0][2]
plaintexts  = metadata['plaintext'][:, 2]
print(f"正确密钥字节：{correct_key}")

# ==================== 标准化 ====================
scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)

# # ==================== 加高斯噪声 ====================
noise_level = 2.0  # 可以改成 0.1, 0.5, 1.0 测试不同强度
#np.random.seed(0)
x_test = x_test + np.random.normal(0, noise_level, x_test.shape)
x_test = x_test.astype(np.float32)  # 加这行
# ==================== AES S-box ====================
AES_Sbox = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

# ==================== LNN 模型定义 ====================
#原版lnn模型
class LNN_SCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(700, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU()
        )
        wiring = AutoNCP(64, 32)
        self.cfc = CfC(64, wiring, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(32, 256))
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.input_proj(x).unsqueeze(1)
        out, _ = self.cfc(x)
        return self.classifier(out[:, -1, :])

#全连接层换成卷积层 再接cfc
# class LNN_SCA(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_proj = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=11, padding=5),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=11, padding=5),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         wiring = AutoNCP(64, 32)
#         self.cfc = CfC(64, wiring, batch_first=True)
#         self.classifier = nn.Sequential(nn.Linear(32, 256))
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.input_proj(x)
#         x = x.permute(0, 2, 1)
#         out, _ = self.cfc(x)
#         return self.classifier(out[:, -1, :])

# 700步模型（CfC直接吃700步，每步1维特征）
# class LNN_SCA(nn.Module):
#     def __init__(self):
#         super().__init__()
#         wiring = AutoNCP(64, 32)
#         self.cfc = CfC(1, wiring, batch_first=True)
#         self.classifier = nn.Linear(32, 256)
#     def forward(self, x):
#         # x: (batch, 700, 1)
#         out, _ = self.cfc(x)
#         return self.classifier(out[:, -1, :])

# ==================== 原版 Rank 计算函数 ====================
def rank(predictions, plaintexts, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        key_bytes_proba = last_key_bytes_proba

    for p in range(max_trace_idx - min_trace_idx):
        plaintext = plaintexts[min_trace_idx + p]
        for i in range(256):
            proba = predictions[p][AES_Sbox[plaintext ^ i]]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                min_proba_predictions = predictions[p][predictions[p] != 0]
                if len(min_proba_predictions) == 0:
                    min_proba = 1e-40
                else:
                    min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba ** 2)

    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba

def full_ranks(predictions, plaintexts, real_key, num_traces=500, rank_step=10):
    index = np.arange(rank_step, num_traces + rank_step, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(len(index))):
        real_key_rank, key_bytes_proba = rank(
            predictions[t-rank_step:t], plaintexts, real_key,
            t-rank_step, t, key_bytes_proba
        )
        f_ranks[i] = [t, real_key_rank]
    return f_ranks

# ==================== 加载模型并预测 ====================
num_traces = 2000

print("\n加载 LNN 模型...")
lnn_model = LNN_SCA().to(device)
lnn_model.load_state_dict(torch.load('lnn_model/lnn_baseline.pth', map_location=device))
#lnn_model.load_state_dict(torch.load('lnn_model/lnn_700steps.pth', map_location=device))
#lnn_model.load_state_dict(torch.load('lnn_model/lnn_700steps_v2.pth', map_location=device))

x_input = torch.tensor(x_test[:num_traces].reshape(-1, 700, 1)).to(device)
with torch.no_grad():
    predictions = torch.softmax(lnn_model(x_input), dim=1).cpu().numpy()

print("计算 Rank...")
ranks = full_ranks(predictions, plaintexts, correct_key, num_traces=num_traces, rank_step=10)

x = [ranks[i][0] for i in range(ranks.shape[0])]
y = [ranks[i][1] for i in range(ranks.shape[0])]

print(f"LNN 最终 Rank: {y[-1]}")

# ==================== 画图 ====================
MODEL_FILE = 'lnn_model/lnn_baseline.pth'
# DATA_FILE = 'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5'

#MODEL_FILE = 'lnn_model/lnn_700steps.pth'
#MODEL_FILE = 'lnn_model/lnn_700steps_v2.pth'
DATA_FILE = 'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5'

first_zero = next((x[i] for i in range(len(y)) if y[i] == 0), None)

plt.plot(x, y)
plt.title(f'Performance of {MODEL_FILE} against {DATA_FILE}')
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)

if first_zero is not None:
    plt.axvline(x=first_zero, color='red', linestyle='--', alpha=0.7)
    plt.text(first_zero + 20, max(y)*0.5, f'Rank=0\n@{first_zero} traces', 
             color='red', fontsize=10)



plt.tight_layout()
plt.savefig('lnn_model/test4.png')

# plt.savefig('lnn_model/rank_lnn_StandardScaler.png')
# print("图已保存到 lnn_model/rank_lnn_StandardScaler.png")

# plt.savefig('lnn_model/lnn_700steps.png')
# print("图已保存到 lnn_model/lnn_700steps.png")
# plt.savefig('lnn_model/lnn_700steps_v2.png')
# print("图已保存到 lnn_model/lnn_700steps_v2.png")

# plt.savefig('lnn_model/rank_lnn_noisy_0.5.png')
# print("图已保存到 lnn_model/rank_lnn_noisy_0.5.png")


# plt.savefig('lnn_model/rank_lnn_noisy_1.0.png')
# print("图已保存到 lnn_model/rank_lnn_noisy_1.0.png")

# plt.savefig('lnn_model/rank_lnn_noisy_2.0.png')
# print("图已保存到 lnn_model/rank_lnn_noisy_2.0.png")