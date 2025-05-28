import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 1: Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2024-12-31")[['Close']]
df.dropna(inplace=True)

# Step 2: Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 60  # 60 days
X, y = create_sequences(scaled_data, SEQ_LEN)

# Step 4: Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 5: Define Dataset & Dataloader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)

# Step 6: Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from last time step
        return out

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train
EPOCHS = 20
for epoch in range(EPOCHS):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Step 8: Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze()
    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 9: Plot
plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('AAPL Stock Price Prediction')
plt.show()
