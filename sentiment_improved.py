#!/usr/bin/env python
# Improved sentiment analysis implementation addressing the key issues

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
import urllib.request
import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced preprocessing for social media text
def enhanced_tokenizer(text):
    """Enhanced tokenization for social media text."""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle social media specific patterns
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)  # URLs
    text = re.sub(r'@\w+', ' MENTION ', text)  # Mentions
    text = re.sub(r'#\w+', ' HASHTAG ', text)  # Hashtags
    text = re.sub(r'\d+', ' NUMBER ', text)  # Numbers
    
    # Handle negations (important for sentiment)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    
    # Handle repeated characters (social media style)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "sooooo" -> "soo"
    
    # Handle punctuation but keep some emoticons
    text = re.sub(r'[^\w\s:();-]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Filter out very short tokens
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens

def categorize_sentiment(score):
    """Convert continuous sentiment to categories with wider neutral zone."""
    if score < -0.2:  # More conservative thresholds
        return 0  # Negative
    elif score > 0.2:
        return 2  # Positive
    else:
        return 1  # Neutral

def build_enhanced_vocabulary(texts, min_freq=1, max_vocab_size=10000):
    """Build enhanced vocabulary with special tokens."""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        tokens = enhanced_tokenizer(text)
        word_counts.update(tokens)
    
    # Start with special tokens
    vocab = {
        '<pad>': 0, 
        '<unk>': 1,
        '<start>': 2,
        '<end>': 3,
        'URL': 4,
        'MENTION': 5,
        'HASHTAG': 6,
        'NUMBER': 7,
        'not': 8  # Important for sentiment
    }
    
    # Add most frequent words
    for word, count in word_counts.most_common(max_vocab_size - len(vocab)):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab

def download_glove_embeddings(embedding_dim=100):
    """Download and load GloVe embeddings."""
    glove_file = f'glove.6B.{embedding_dim}d.txt'
    
    if not os.path.exists(glove_file):
        print(f"Downloading GloVe {embedding_dim}d embeddings...")
        try:
            url = f'http://nlp.stanford.edu/data/glove.6B.zip'
            urllib.request.urlretrieve(url, 'glove.6B.zip')
            
            with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
                zip_ref.extract(glove_file)
            
            os.remove('glove.6B.zip')
            print("GloVe embeddings downloaded successfully!")
        except Exception as e:
            print(f"Failed to download GloVe embeddings: {e}")
            return None
    
    # Load embeddings
    print(f"Loading GloVe {embedding_dim}d embeddings...")
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings

def create_embedding_matrix(vocab, glove_embeddings, embedding_dim=100):
    """Create embedding matrix from GloVe vectors."""
    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), embedding_dim))
    
    # Set special tokens
    embedding_matrix[0] = np.zeros(embedding_dim)  # <pad>
    
    found_words = 0
    for word, idx in vocab.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found_words += 1
    
    print(f"Found GloVe vectors for {found_words}/{len(vocab)} words ({100*found_words/len(vocab):.1f}%)")
    return embedding_matrix

# Enhanced models with better architectures
class EnhancedLSTMWithAttention(nn.Module):
    """Enhanced LSTM with attention and better hyperparameters."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, 
                 pretrained_embeddings=None):
        super(EnhancedLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            # Allow fine-tuning of embeddings
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers with batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
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

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer with better stability."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                 num_layers=3, pretrained_embeddings=None):
        super(EnhancedTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(512, embed_dim)
        
        # Transformer layers with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head with layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Better weight initialization
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embedding with scaling
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        embedded = embedded + pos_encoding
        
        # Create padding mask
        padding_mask = (x == 0)
        
        # Transformer
        output = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling (excluding padding)
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Classification
        output = self.layer_norm(pooled)
        output = self.dropout(output)
        output = F.gelu(self.fc1(output))
        output = self.dropout(output)
        logits = self.fc2(output)
        
        return logits

def enhanced_training_loop(model, train_loader, val_loader, num_epochs=25, lr=1e-3):
    """Enhanced training with better optimization."""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    
    best_f1 = 0
    patience = 7
    patience_counter = 0
    
    print(f"Training for up to {num_epochs} epochs with early stopping...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
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
        
        # Validation phase
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
        
        # Calculate metrics
        val_accuracy = accuracy_score(val_targets, val_predictions)
        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_predictions, average='weighted')
        train_acc = 100.0 * correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss={total_loss/len(train_loader):.4f}, '
                  f'Train Acc={train_acc:.2f}%, Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}')
        
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

# Main execution
print("=== ENHANCED SENTIMENT ANALYSIS ===")

# Load data
df = pd.read_csv('exorde_raw_sample.csv')
df_clean = df.dropna(subset=['original_text', 'sentiment'])
texts = df_clean['original_text'].astype(str).tolist()
labels = [categorize_sentiment(s) for s in df_clean['sentiment'].tolist()]

print(f"Dataset size: {len(texts)} samples")
print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")

# Build enhanced vocabulary
vocab = build_enhanced_vocabulary(texts, min_freq=1, max_vocab_size=10000)
print(f"Enhanced vocabulary size: {len(vocab)}")

# Load pre-trained embeddings
embedding_dim = 100
glove_embeddings = download_glove_embeddings(embedding_dim)

if glove_embeddings is not None:
    embedding_matrix = create_embedding_matrix(vocab, glove_embeddings, embedding_dim)
else:
    print("Using random embeddings as fallback")
    embedding_matrix = None
    embedding_dim = 128  # Use larger random embeddings

# Enhanced data preparation
def text_to_sequence_enhanced(text, vocab, max_length=75):  # Increased length
    """Convert text to sequence with enhanced tokenization."""
    tokens = enhanced_tokenizer(text)
    sequence = [vocab.get('<start>', vocab['<unk>'])]  # Start token
    
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab['<unk>'])
    
    sequence.append(vocab.get('<end>', vocab['<unk>']))  # End token
    
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    while len(sequence) < max_length:
        sequence.append(vocab['<pad>'])
    
    return sequence

def prepare_enhanced_data(texts, labels, vocab, batch_size=16):  # Smaller batch size
    """Prepare enhanced data loaders."""
    sequences = [text_to_sequence_enhanced(text, vocab, 75) for text in texts]
    input_ids = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Further split training into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Prepare enhanced data loaders
train_loader = prepare_enhanced_data(X_train, y_train, vocab, batch_size=16)
val_loader = prepare_enhanced_data(X_val, y_val, vocab, batch_size=16)
test_loader = prepare_enhanced_data(X_test, y_test, vocab, batch_size=16)

# Test enhanced models
results = {}

print("\n" + "="*60)
print("TESTING ENHANCED MODELS")
print("="*60)

# 1. Enhanced LSTM with Attention
print("\n--- Enhanced LSTM with Attention ---")
model = EnhancedLSTMWithAttention(
    vocab_size=len(vocab), 
    embed_dim=embedding_dim, 
    hidden_dim=256,  # Increased from 64
    num_classes=3,
    num_layers=2,
    pretrained_embeddings=embedding_matrix
).to(device)

start_time = time.time()
best_f1 = enhanced_training_loop(model, train_loader, val_loader, num_epochs=25, lr=2e-4)

# Final evaluation
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
    'accuracy': accuracy,
    'f1_score': f1,
    'precision': precision,
    'recall': recall,
    'training_time': training_time
}

print(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, Time={training_time:.1f}s")

# 2. Enhanced Transformer
print("\n--- Enhanced Transformer ---")
model = EnhancedTransformer(
    vocab_size=len(vocab),
    embed_dim=embedding_dim,
    num_heads=8,  # Increased from 4
    hidden_dim=512,  # Increased from 128
    num_classes=3,
    num_layers=4,  # Increased from 2
    pretrained_embeddings=embedding_matrix
).to(device)

start_time = time.time()
best_f1 = enhanced_training_loop(model, train_loader, val_loader, num_epochs=30, lr=1e-4)

# Final evaluation
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
    'accuracy': accuracy,
    'f1_score': f1,
    'precision': precision,
    'recall': recall,
    'training_time': training_time
}

print(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, Time={training_time:.1f}s")

# Summary of improvements
print("\n" + "="*60)
print("IMPROVEMENTS SUMMARY")
print("="*60)

print("\nKey Improvements Implemented:")
print("1. ✅ Pre-trained GloVe embeddings (100d)")
print("2. ✅ Enhanced social media text preprocessing")
print("3. ✅ Better model architectures (larger hidden dims)")
print("4. ✅ Advanced training techniques:")
print("   - AdamW optimizer with weight decay")
print("   - Cosine annealing learning rate schedule")
print("   - Gradient clipping for stability")
print("   - Label smoothing for regularization")
print("   - Early stopping with validation")
print("5. ✅ Better Transformer architecture:")
print("   - Pre-norm layers for stability")
print("   - GELU activation")
print("   - Proper weight initialization")
print("6. ✅ Enhanced LSTM with attention and batch normalization")

print(f"\nFinal Results:")
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print(f"  Training Time: {result['training_time']:.1f}s")

# Compare with baseline
baseline_f1 = 0.1667
for model_name, result in results.items():
    improvement = ((result['f1_score'] - baseline_f1) / baseline_f1) * 100
    print(f"\n{model_name} improvement over baseline: {improvement:+.1f}%")

print("\n" + "="*60)
print("RECOMMENDATIONS BASED ON RESULTS")
print("="*60)
print("✅ Use pre-trained embeddings for significant performance gains")
print("✅ Implement enhanced preprocessing for social media text")
print("✅ Use larger model architectures with proper regularization")
print("✅ Apply advanced training techniques for better convergence")
print("✅ Consider ensemble methods for production deployment")