# movie-review-analysis
WiSe 25/26 Advanced NLP Project: NLP pipeline for sentiment analysis on IMDB reviews

```
movie-review-analysis/
├── config.yaml                  # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main pipeline script
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading and exploration
│   │   └── preprocessing.py    # Text preprocessing (cleaning up+tokenization)
│   ├── models/
│   │   ├── ngram.py            # N-gram models
│   │   ├── nn.py               # LSTM implementation
│   │   └── transformer.py      # Transformer fine-tuning
│   └──evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── llm_judge.py        # LLM and human evaluation
└── dataset/                    
    └── imdb-dataset.csv        # IMDB Dataset of 50K Movie Reviews (ignored)

```