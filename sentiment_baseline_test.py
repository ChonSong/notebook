#!/usr/bin/env python
# Quick baseline test to check current performance before implementing improvements

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load existing dataset
df = pd.read_csv('exorde_raw_sample.csv')
print(f"Dataset loaded: {len(df)} samples")

# Simple preprocessing functions
def simple_tokenizer(text):
    """Basic tokenization and cleaning."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def categorize_sentiment(score):
    """Convert continuous sentiment to categories."""
    if score < -0.1:
        return 0  # Negative
    elif score > 0.1:
        return 2  # Positive
    else:
        return 1  # Neutral

def build_vocabulary(texts, min_freq=2, max_vocab_size=5000):
    """Build vocabulary from texts."""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        tokens = simple_tokenizer(text)
        word_counts.update(tokens)
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def text_to_sequence(text, vocab, max_length=50):
    """Convert text to sequence of token IDs."""
    tokens = simple_tokenizer(text)
    sequence = []
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab['<unk>'])
    
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    while len(sequence) < max_length:
        sequence.append(vocab['<pad>'])
    
    return sequence

def tokenize_texts(texts, vocab, max_length=50):
    """Tokenize list of texts."""
    sequences = [text_to_sequence(text, vocab, max_length) for text in texts]
    return torch.tensor(sequences, dtype=torch.long)

def prepare_data(texts, labels, vocab, batch_size=32):
    """Prepare data loaders."""
    input_ids = tokenize_texts(texts, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Simple LSTM model for baseline test
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.dropout(hidden[-1])
        logits = self.fc(output)
        return logits

# Test baseline performance
print("\n=== BASELINE PERFORMANCE TEST ===")

# Prepare data
df_clean = df.dropna(subset=['original_text', 'sentiment'])
texts = df_clean['original_text'].astype(str).tolist()
labels = [categorize_sentiment(s) for s in df_clean['sentiment'].tolist()]

print(f"Dataset size: {len(texts)} samples")
print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")

# Build vocabulary  
vocab = build_vocabulary(texts, min_freq=2, max_vocab_size=5000)
print(f"Vocabulary size: {len(vocab)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Prepare data loaders
train_loader = prepare_data(X_train, y_train, vocab, batch_size=32)
test_loader = prepare_data(X_test, y_test, vocab, batch_size=32)

# Train baseline model
model = SimpleLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=3)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("\nTraining baseline LSTM for 10 epochs...")
start_time = time.time()

# Simple training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    if (epoch + 1) % 3 == 0:
        acc = 100.0 * correct / total
        print(f'Epoch {epoch+1}/10: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%')

# Evaluate model
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_targets, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
training_time = time.time() - start_time

print(f"\n=== BASELINE RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Training Time: {training_time:.1f}s")

# Show current hyperparameters
print(f"\n=== CURRENT HYPERPARAMETERS ===")
print(f"Embedding dimension: 64")
print(f"Hidden dimension: 64")
print(f"Learning rate: 1e-3")
print(f"Epochs: 10")
print(f"Vocabulary size: {len(vocab)}")
print(f"Max sequence length: 50")

print("\nBaseline test completed. These are the performance levels we need to improve upon.")