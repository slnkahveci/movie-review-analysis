# IMDB Sentiment Analysis

Multi-model sentiment analysis project with N-grams, RNNs, and Transformers.


## Project Structure
```
src/
  ├── models/
  │   ├── ngram.py                              # N-gram language models (bigram/trigram) with Laplace smoothing
  │   ├── nn.py                                 # LSTM/GRU implementations with dataset classes
  │   └── transformer.py                        # ALBERT fine-tuning for sentiment classification
  ├── data/
  │   ├── dataloader.py                         # IMDB dataset loading and preprocessing
  │   └── stats.py                              # Dataset statistics and visualizations
  └── eval/
      ├── metrics.py                            # Evaluation metrics (accuracy, F1, perplexity)
      ├── llm_judge.py                          # LLM-as-a-Judge evaluation implementation
      └── eval_transformer.py                   # Main evaluation pipeline script
out/                                            # (ignored) All outputs (models, plots, results)
  ├── wordclouds.png                            # Word cloud visualizations for positive/negative reviews
  ├── scattertext.html                          # Interactive term frequency visualization
  ├── minilm_imdb_model/                        # Fine-tuned transformer model directory
  └── evaluation_results.csv                    # Evaluation results for manual annotation
dataset/                                        # (ignored) IMDB dataset
  ├── imdb-dataset.csv                          # Original IMDB dataset
  └── imdb-test-subsample-100_{SAMPLE_SIZE}.csv # Random 100 test samples for evaluation
interface.ipynb                                 # Unified Jupyter interface for training and evaluation
requirements.txt
```

## Models

### N-gram Models (`src/models/ngram.py`)
- Bigram and trigram language models
- Laplace smoothing for probability estimation
- Text generation and perplexity computation

### RNN Models (`src/models/nn.py`)
- LSTM and GRU implementations
- Custom `RNNDataset` and `RNNDataModule`
- Vocabulary building and text generation

### Transformer Models (`src/models/transformer.py`)
- Fine-tuning ALBERT on sentiment classification
- Custom `TransformerDataset` for pre-tokenized data
- Support for multiple tokenization strategies (WordPiece, BPE, Unigram)

## Evaluation

The evaluation pipeline (`src/eval/eval_transformer.py`) implements three approaches:

1. **Metric-based Evaluation** (with gold labels)
   - Accuracy, F1 score, Perplexity, Confusion matrix
   - Evaluated on train/val/test splits + 100 random subsample

2. **Human Evaluation** (simulates no gold labels)
   - Manual rating by you and teammates
   - CSV-based annotation workflow

3. **LLM-as-a-Judge** (automated evaluation)

## Output Files

All outputs are saved in the `out/` directory: 

- `out/lstm_imdb_model.pth` - Trained LSTM model
- `out/gru_imdb_model.pth` - Trained GRU model
- `out/minilm_imdb_model/` - Fine-tuned transformer
- `out/wordclouds.png` - Word frequency visualizations
- `out/scattertext.html` - Interactive term explorer
- `out/model_comparison.png` - Perplexity comparison
- `out/evaluation_results.csv` - Evaluation results

## Requirements

- Python 3.8+ (used 3.9.20)
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy
- matplotlib, wordcloud, scattertext
- tqdm, nltk, requests
- OpenRouter API key or Gemini API key

