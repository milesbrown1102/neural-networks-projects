import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os

# Load and preprocess data
df = pd.read_csv(os.path.expanduser("~/Projects/RNN/IMDB-Dataset.csv"), names=["text", "label"])
df['text'] = df['text'].str.lower().str.split()

# Filter only 'positive' and 'negative' labels if needed
df = df[df['label'].isin(['positive', 'negative'])]

# Encode labels (0 and 1 only)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # positive → 1, negative → 0

# Split data
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Build vocabulary
vocab = {word for phrase in df['text'] for word in phrase}
word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}

# Pad and encode
max_length = df['text'].str.len().max()

def encode_and_pad(text):
    encoded = [word_to_idx.get(word, 0) for word in text]
    return encoded + [0] * (max_length - len(encoded))

train_data['text'] = train_data['text'].apply(encode_and_pad)
test_data['text'] = test_data['text'].apply(encode_and_pad)

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = torch.tensor(data['text'].to_list(), dtype=torch.long)
        self.labels = torch.tensor(data['label'].to_list(), dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# DataLoaders
train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Model config
vocab_size = len(vocab) + 1
embed_size = 128
hidden_size = 128
output_size = 2

model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)

# Training config
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for texts, labels in train_loader:
        outputs = model(texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
