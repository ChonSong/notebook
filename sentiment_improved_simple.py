#!/usr/bin/env python
# Simplified improved sentiment analysis focusing on core enhancements

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced preprocessing for social media text
def enhanced_tokenizer(text):
    """Enhanced tokenization for social media text."""
    import re
    
    text = text.lower()
    
    # Handle social media specific patterns
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' MENTION ', text) 
    text = re.sub(r'#\w+', ' HASHTAG ', text)
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # Handle negations (crucial for sentiment)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    
    # Handle repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Clean punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens

def categorize_sentiment(score):
    """More conservative sentiment categorization."""
    if score < -0.2:  # Wider neutral zone
        return 0  # Negative
    elif score > 0.2:
        return 2  # Positive
    else:
        return 1  # Neutral

def build_enhanced_vocabulary(texts, min_freq=1, max_vocab_size=5000):
    """Build enhanced vocabulary."""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        tokens = enhanced_tokenizer(text)
        word_counts.update(tokens)
    
    vocab = {
        '<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3,
        'URL': 4, 'MENTION': 5, 'HASHTAG': 6, 'NUMBER': 7,
        'not': 8, 'good': 9, 'bad': 10, 'great': 11, 'terrible': 12
    }
    
    for word, count in word_counts.most_common(max_vocab_size - len(vocab)):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab

# Enhanced LSTM with Attention
class EnhancedLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super(EnhancedLSTMWithAttention, self).__init__()
        
        # Larger embedding with proper initialization
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        # Bidirectional LSTM with dropout
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers with batch norm
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.bn1(attended_output)
        output = self.dropout1(output)
        output = F.relu(self.fc1(output))
        output = self.bn2(output)
        output = self.dropout2(output)
        logits = self.fc2(output)
        
        return logits

# Enhanced Transformer
class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=3):
        super(EnhancedTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(100, embed_dim)
        
        # Stable transformer with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.2, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embedding with scaling
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        embedded = embedded + pos_encoding
        
        # Padding mask
        padding_mask = (x == 0)
        
        # Transformer
        output = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Classification
        output = self.layer_norm(pooled)
        output = self.dropout(output)
        output = F.gelu(self.fc1(output))
        output = self.dropout(output)
        logits = self.fc2(output)
        
        return logits

def enhanced_training_loop(model, train_loader, val_loader, num_epochs=20, lr=1e-3):
    """Enhanced training with better optimization."""
    
    # Better optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, eta_min=1e-6
    )
    
    # Label smoothing for regularization
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_accuracy = accuracy_score(val_targets, val_predictions)
        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_predictions, average='weighted')
        train_acc = 100.0 * correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, '
                  f'Train Acc={train_acc:.2f}%, Val F1={val_f1:.4f}')
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return best_f1

# Enhanced data preparation
def text_to_sequence_enhanced(text, vocab, max_length=60):
    tokens = enhanced_tokenizer(text)
    sequence = [vocab.get('<start>', vocab['<unk>'])]
    
    for token in tokens:
        sequence.append(vocab.get(token, vocab['<unk>']))
    
    sequence.append(vocab.get('<end>', vocab['<unk>']))
    
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    while len(sequence) < max_length:
        sequence.append(vocab['<pad>'])
    
    return sequence

def prepare_enhanced_data(texts, labels, vocab, batch_size=16):
    sequences = [text_to_sequence_enhanced(text, vocab) for text in texts]
    input_ids = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Main execution
print("=== ENHANCED SENTIMENT ANALYSIS (SIMPLIFIED) ===")

# Load and prepare data
df = pd.read_csv('exorde_raw_sample.csv')
df_clean = df.dropna(subset=['original_text', 'sentiment'])
texts = df_clean['original_text'].astype(str).tolist()
labels = [categorize_sentiment(s) for s in df_clean['sentiment'].tolist()]

print(f"Dataset: {len(texts)} samples")
print(f"Labels: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")

# Enhanced vocabulary
vocab = build_enhanced_vocabulary(texts, min_freq=1, max_vocab_size=5000)
print(f"Enhanced vocabulary: {len(vocab)} tokens")

# Data splits
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Data loaders
train_loader = prepare_enhanced_data(X_train, y_train, vocab, 16)
val_loader = prepare_enhanced_data(X_val, y_val, vocab, 16)
test_loader = prepare_enhanced_data(X_test, y_test, vocab, 16)

results = {}

# Test Enhanced LSTM with Attention
print("\n" + "="*50)
print("ENHANCED LSTM WITH ATTENTION")
print("="*50)

model = EnhancedLSTMWithAttention(
    vocab_size=len(vocab), 
    embed_dim=128,  # Larger than baseline 64
    hidden_dim=256,  # Much larger than baseline 64
    num_classes=3,
    num_layers=2
).to(device)

start_time = time.time()
enhanced_training_loop(model, train_loader, val_loader, num_epochs=20, lr=2e-4)

# Test evaluation
model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

accuracy = accuracy_score(test_targets, test_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='weighted')
training_time = time.time() - start_time

results['Enhanced_LSTM_Attention'] = {
    'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 
    'recall': recall, 'training_time': training_time
}

print(f"Enhanced LSTM Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Time: {training_time:.1f}s")

# Test Enhanced Transformer
print("\n" + "="*50)
print("ENHANCED TRANSFORMER")
print("="*50)

model = EnhancedTransformer(
    vocab_size=len(vocab),
    embed_dim=128,  # Larger than baseline
    num_heads=8,    # More heads than baseline 4
    hidden_dim=512, # Much larger than baseline 128
    num_classes=3,
    num_layers=4    # More layers than baseline 2
).to(device)

start_time = time.time()
enhanced_training_loop(model, train_loader, val_loader, num_epochs=25, lr=1e-4)

# Test evaluation
model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

accuracy = accuracy_score(test_targets, test_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='weighted')
training_time = time.time() - start_time

results['Enhanced_Transformer'] = {
    'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 
    'recall': recall, 'training_time': training_time
}

print(f"Enhanced Transformer Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Time: {training_time:.1f}s")

# Summary
print("\n" + "="*60)
print("IMPROVEMENT SUMMARY")
print("="*60)

baseline_accuracy = 0.3333
baseline_f1 = 0.1667

print("Key Improvements Applied:")
print("✅ Enhanced social media text preprocessing")
print("✅ Larger model architectures (embed_dim: 64→128, hidden_dim: 64→256/512)")
print("✅ Better training techniques:")
print("   - AdamW optimizer with weight decay")
print("   - Cosine annealing scheduler")
print("   - Gradient clipping")
print("   - Label smoothing")
print("   - Early stopping")
print("✅ Improved Transformer with stability fixes")
print("✅ Attention mechanism with batch normalization")

print(f"\nPerformance Comparison:")
print(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10} {'Improvement':<12}")
print("-" * 65)

print(f"{'Baseline LSTM':<25} {baseline_accuracy:<10.4f} {baseline_f1:<10.4f} {'--':<12}")

for model_name, result in results.items():
    acc_improvement = ((result['accuracy'] - baseline_accuracy) / baseline_accuracy) * 100
    f1_improvement = ((result['f1_score'] - baseline_f1) / baseline_f1) * 100
    print(f"{model_name:<25} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {f1_improvement:+8.1f}%")

print("\nRecommendations:")
print("1. Use enhanced preprocessing for social media text")
print("2. Increase model capacity (larger embedding/hidden dimensions)")
print("3. Apply modern training techniques for better convergence")
print("4. Consider pre-trained embeddings for further improvements")
print("5. Use ensemble methods for production deployment")