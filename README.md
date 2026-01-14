# IMDB Sentiment Analysis

Multi-model sentiment analysis project with N-grams, RNNs, and Transformers.


## Project Structure

```
src/
  ├── models/       # Model implementations (ngram, nn, transformer)
  ├── data/         # Data processing (dataloader, stats)
  └── eval/         # Evaluation scripts (metrics, LLM judge)
out/                # (ignored) All outputs (models, plots, results) 
dataset/            # (ignored) IMDB dataset
interface.ipynb     # Unified Jupyter interface
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

