import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import matplotlib.pyplot as plt

# ==================== 資料讀取與前處理 ====================
df = pd.read_csv("data/2317_short_term_with_lag3.csv").iloc[:, :6].copy()

# 保留日期欄
dates = df.iloc[:, 0]

# 對其餘欄位標準化 TODO:考慮要不要標準化
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df.iloc[:, 1:])  # 只對數值欄做標準化

# 合併回 DataFrame，保留欄位名稱
scaled_df = pd.DataFrame(scaled_values, columns=df.columns[1:])
scaled_df.insert(0, df.columns[0], dates)
# 計算漲跌：今天收盤價 vs 昨天收盤價
scaled_df['漲跌'] = scaled_df['收盤價(元)'].diff().apply(lambda x: 1 if x > 0 else -1)
scaled_df= scaled_df.iloc[1:].reset_index(drop=True)


# 結果 DataFrame：日期 + 標準化後數值
print(scaled_df.head())
SEQ_LEN = 10
PRED_IDX = 0  # 預測第 0 欄（可改）
print(scaled_df['漲跌'].value_counts())

scaled_df['漲跌'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('漲跌分布統計')
plt.xlabel('漲跌')
plt.ylabel('天數')
plt.show()



# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, seq_len=SEQ_LEN):
#         self.x = []
#         self.y = []
#         for i in range(len(data) - seq_len):
#             self.x.append(data[i:i+seq_len])
#             self.y.append(data[i+seq_len, PRED_IDX])
#         self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
#         self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

# # DataLoader
# batch_size = 32
# dataset = TimeSeriesDataset(scaled_data)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # ==================== Transformer 模型 ====================
# class TransformerModel(nn.Module):
#     def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(d_model, 1)

#     def forward(self, x):
#         x = self.input_proj(x)               # (B, T, d_model)
#         x = self.transformer_encoder(x)      # (B, T, d_model)
#         out = x[:, -1, :]                    # 最後一個時間步
#         return self.fc_out(out)              # (B, 1)

# # ==================== 模型訓練 ====================
# model = TransformerModel()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for x_batch, y_batch in dataloader:
#         optimizer.zero_grad()
#         output = model(x_batch)
#         loss = criterion(output, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# # ==================== 推論（預測下一筆） ====================
# model.eval()
# with torch.no_grad():
#     last_seq = torch.tensor(scaled_data[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
#     prediction = model(last_seq)
#     print("下一筆預測（scaled）:", prediction.item())