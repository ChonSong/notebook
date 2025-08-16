# Comprehensive Sentiment Analysis Notebook

This repository contains a comprehensive sentiment analysis notebook that implements multiple deep learning architectures based on recent literature in natural language processing.

## Overview

The notebook implements various models for sentiment analysis, including:

1. **FastText** - Simple but effective baseline (Joulin et al., 2016)
2. **RNN/LSTM/GRU** - Basic sequential models
3. **Bidirectional RNNs** - Process sequences in both directions (Huang, Xu, and Yu, 2015)
4. **Attention-based Models** - Focus on important words (Lin et al., 2017)
5. **Transformer** - Self-attention mechanism (Vaswani et al., 2017)

## Literature References

The models implemented are based on the following papers:

1. **Vaswani et al. (2017)** - "Attention Is All You Need" - Transformer architecture
2. **Huang, Xu, and Yu (2015)** - "Bidirectional LSTM-CRF Models for Sequence Tagging" - Bi-LSTM
3. **Lin et al. (2017)** - "A Structured Self-Attentive Sentence Embedding" - Self-attention
4. **Pennington, Socher, and Manning (2014)** - "GloVe: Global Vectors for Word Representation" - Word embeddings
5. **Joulin et al. (2016)** - "Bag of Tricks for Efficient Text Classification" - FastText

## Usage

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Open `sentiment_analysis_comprehensive.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially - the notebook is designed to run in isolation
3. The notebook will automatically download the required CSV data in the "getdata" cell
4. All models will be trained and compared automatically

### Features

- **Self-contained**: Downloads data automatically, no external files needed except CSV data
- **Comprehensive**: Implements 11 different model architectures
- **Well-documented**: Detailed comments explaining each approach and referencing relevant papers
- **Comparative**: Evaluates all models and provides performance comparisons
- **Reproducible**: Set random seeds for consistent results

### Output

The notebook produces:
- Training curves for all models
- Performance comparison charts
- Confusion matrices
- Detailed analysis relating results back to the literature
- Saved model results for future reference

### File Structure

```
notebook/
├── sentiment_analysis_comprehensive.ipynb  # Main notebook
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
└── data/                                 # Created automatically
    └── IMDB Dataset.csv                  # Downloaded automatically
```

## Model Performance

The notebook trains and compares multiple architectures, providing insights into:
- Which architectures work best for sentiment analysis
- How attention mechanisms improve performance
- The effectiveness of bidirectional processing
- Trade-offs between model complexity and performance

## Notes

- The notebook is designed to run sequentially without errors
- All dependencies are handled within the notebook
- Data is downloaded automatically in the appropriate cell
- Results are saved for future reference
- Comments throughout reference the specific papers and techniques used