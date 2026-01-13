"""
IMDB Dataset Loader for Data Exploration and Model Training
Supports: data exploration, n-gram models, neural network training
"""

import pandas as pd
import numpy as np
import contractions
import functools
from collections import Counter
from typing import Optional, Literal, Callable
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch
from torch.utils.data import Dataset, DataLoader


# Shared mapping from sentiment string to numeric id
SENTIMENT_TO_ID = {"negative": 0, "positive": 1}


# ==================== Part 1: Preprocessing and Persistence ====================

class TextPreprocessor:
    """
    Handles data loading, preprocessing, and persistence for IMDB dataset.
    
    Usage:
        preprocessor = TextPreprocessor("dataset/imdb-dataset.csv")
        df = preprocessor.load_data(remove_stopwords=True)
        train_df, val_df, test_df = preprocessor.get_splits()
        preprocessor.save("processed_data")
    """

    def __init__(
        self, file_path: str, sample_size: Optional[int] = 50000, random_state: int = 42, tokenizer_type: Literal["word", "bpe", "wordpiece"] = "word"
    ):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.random_state = random_state

        self.data: Optional[pd.DataFrame] = None

        self.stop_words = set(stopwords.words("english"))
        self._tokenizer = (
            functools.partial(nltk.word_tokenize, language="english")
            if tokenizer_type == "word"
            else None
        )

    def _preprocess_text(self, df: pd.DataFrame, remove_stopwords: bool = False, stemming: Literal["none", "stemming", "lemmatization"] = "none") -> pd.DataFrame:
        """Basic text preprocessing: lowercasing and removing punctuation."""
        text_column = df["review"]
        # Clean HTML tags
        text_column = text_column.str.replace(r"<br\s*/?>", " ", regex=True)
        # Lowercasing
        text_column = text_column.str.lower()

        # Expand contractions
        text_column = text_column.apply(contractions.fix)

        # Remove stopwords
        if remove_stopwords:
            text_column = text_column.apply(
                lambda x: " ".join([word for word in x.split() if word not in self.stop_words])
            )

        # Remove punctuation and numbers
        text_column = text_column.str.replace(r"[^\w\s!?]", "", regex=True) 

        # Lemmatization and Stemming
        if stemming == "lemmatization":
            lemmatizer = nltk.WordNetLemmatizer()
            text_column = text_column.apply(
                lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
            )
        elif stemming == "stemming":
            stemmer = nltk.PorterStemmer()
            text_column = text_column.apply(
                lambda x: " ".join([stemmer.stem(word) for word in x.split()])
            )
        return text_column

    def _tokenize_words(self, text: str) -> list[str]:
        return self._tokenizer(text)

    def load_data(self, remove_stopwords: bool = False) -> pd.DataFrame:
        # Load and preprocess the IMDB dataset.

        print(f"Loading data from {self.file_path}...")

        df = pd.read_csv(self.file_path)

        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=self.random_state)
            df = df.reset_index(drop=True)

        df["review"] = self._preprocess_text(df, remove_stopwords)
        df["_tokens"] = df["review"].apply(self._tokenize_words)

        # Length features
        df["_char_len"] = df["review"].str.len()
        df["_word_count"] = df["_tokens"].apply(len)

        self.data = df

        sentiment_counts = df["sentiment"].value_counts().to_dict()
        print(
            f"Loaded {len(df)} reviews (positive: {sentiment_counts.get('positive', 0)}, negative: {sentiment_counts.get('negative', 0)})"
        )
        return df

    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.

        Returns:
            (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1"

        from sklearn.model_selection import train_test_split

        stratify_col = self.data["sentiment"] if stratify else None

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            self.data,
            train_size=train_ratio,
            stratify=stratify_col,
            random_state=self.random_state,
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify_col = temp_df["sentiment"] if stratify else None

        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            stratify=stratify_col,
            random_state=self.random_state,
        )

        print( f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # reset indices and don't save old indices
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def save(self, path: str):
        """Save processed data to disk."""
        if self.data is None:
            raise ValueError("No data to save.")

        from datasets import Dataset

        # Convert to HF Dataset (exclude non-serializable columns)
        save_cols = [
            "review",
            "sentiment",
            "_char_len",
            "_word_count",
        ]
        save_df = self.data[save_cols].copy()

        # Store tokens as strings (HF doesn't like nested lists well)
        save_df["_tokens_str"] = self.data["_tokens"].apply(" ".join)

        Dataset.from_pandas(save_df).save_to_disk(path)
        print(f"Saved to {path}")

    def load(self, path: str) -> pd.DataFrame:
        """Load processed data from disk."""
        from datasets import load_from_disk

        df = load_from_disk(path).to_pandas()

        # Reconstruct tokens
        df["_tokens"] = df["_tokens_str"].str.split()
        df = df.drop(columns=["_tokens_str"])

        self.data = df

        print(f"Loaded {len(df)} reviews from {path}")
        return df


# ==================== Part 2: Statistics ====================

class IMDBDataStats:
    """
    Computes statistics and visualizations for IMDB dataset.
    
    Usage:
        preprocessor = IMDBDataPreprocessor("dataset/imdb-dataset.csv")
        df = preprocessor.load_data()
        
        stats_computer = IMDBDataStats(df)
        stats = stats_computer.get_stats()
        stats_computer.visualize_word_clouds()
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a preprocessed DataFrame.
        
        Args:
            data: DataFrame from IMDBDataPreprocessor.load_data()
        """
        if data is None or "_tokens" not in data.columns:
            raise ValueError(
                "Data must be a DataFrame with '_tokens' column. Use IMDBDataPreprocessor.load_data() first."
            )
        self.data = data
        self._word_counters: Optional[dict[str, Counter]] = None
        self.corpus = None  # scattertext corpus (lazy)
        self.stats: Optional[dict] = None

    def _get_word_counters(self) -> dict[str, Counter]:
        """Get word frequency counters per sentiment (cached)."""
        if self._word_counters is not None:
            return self._word_counters

        self._word_counters = {}
        for sentiment in ["positive", "negative"]:
            mask = self.data["sentiment"] == sentiment
            tokens = self.data.loc[mask, "_tokens"].explode()
            self._word_counters[sentiment] = Counter(tokens.dropna())

        all_tokens = self.data["_tokens"].explode()
        self._word_counters["overall"] = Counter(all_tokens.dropna())

        return self._word_counters

    def _build_corpus(self):
        """Build scattertext corpus (lazy)."""
        if self.corpus is not None:
            return self.corpus

        import scattertext as st

        print("Building scattertext corpus...")
        self.corpus = st.CorpusFromPandas(
            self.data,
            category_col="sentiment",
            text_col="review",
            nlp=st.whitespace_nlp_with_sentences,
        ).build()

        return self.corpus

    def get_simple_stats(self) -> dict:
        """Compute basic dataset statistics."""
        stats = {}

        # Class distribution
        stats["class_distribution"] = self.data["sentiment"].value_counts().to_dict()

        # Length statistics
        stats["average_text_length"] = {
            "overall": self.data["_char_len"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_char_len"
            ].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_char_len"
            ].mean(),
        }
        stats["median_text_length"] = {
            "overall": self.data["_char_len"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_char_len"
            ].median(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_char_len"
            ].median(),
        }
        stats["average_word_count"] = {
            "overall": self.data["_word_count"].mean(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_word_count"
            ].mean(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_word_count"
            ].mean(),
        }
        stats["median_word_count"] = {
            "overall": self.data["_word_count"].median(),
            "positive": self.data[self.data["sentiment"] == "positive"][
                "_word_count"
            ].median(),
            "negative": self.data[self.data["sentiment"] == "negative"][
                "_word_count"
            ].median(),
        }

        # Vocabulary
        counters = self._get_word_counters()
        stats["vocabulary_size"] = {k: len(v) for k, v in counters.items()}
        stats["most_frequent_words"] = {
            k: v.most_common(10) for k, v in counters.items()
        }
        self.stats = stats
        return stats

    def get_full_stats(self) -> dict:
        """Compute comprehensive dataset statistics."""
        stats = self.get_simple_stats()
        # N-grams
        for n in [2, 3, 4]:
            stats[f"{n}_gram_frequency"] = self._compute_ngram_stats(n)

        # Log-odds ratio
        stats["most_significant_words"] = self._compute_log_odds_ratio()

        self.stats = stats
        return stats

    def _compute_ngram_stats(self, n: int) -> dict[str, list]:
        """Compute n-gram frequencies per sentiment."""
        result = {}

        for sentiment in ["positive", "negative", "overall"]:
            if sentiment == "overall":
                tokens_series = self.data["_tokens"]
            else:
                mask = self.data["sentiment"] == sentiment
                tokens_series = self.data.loc[mask, "_tokens"]

            ngram_counter = Counter()
            for tokens in tokens_series:
                ngram_counter.update(nltk.ngrams(tokens, n))

            result[sentiment] = ngram_counter.most_common(10)

        return result

    def _compute_log_odds_ratio(self, top_n: int = 10) -> dict[str, list]:
        """Compute log-odds ratio scores using scattertext."""
        from scattertext import LogOddsRatioUninformativeDirichletPrior

        corpus = self._build_corpus()
        term_freq_df = corpus.get_term_freq_df()

        scorer = LogOddsRatioUninformativeDirichletPrior()
        scores = scorer.get_scores(
            term_freq_df["positive freq"], term_freq_df["negative freq"]
        )

        scores_series = pd.Series(scores, index=term_freq_df.index)

        return {
            "positive": scores_series.nlargest(top_n).index.tolist(),
            "negative": scores_series.nsmallest(top_n).index.tolist(),
        }

    def visualize_word_clouds(self, save_path: Optional[str] = None):
        """Generate word clouds per sentiment."""
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        counters = self._get_word_counters()

        for ax, sentiment in zip(axes, ["positive", "negative"]):
            word_freq = {
                w: c
                for w, c in counters[sentiment].items()
                if w not in STOPWORDS and len(w) > 2
            }

            wc = WordCloud(
                width=800, height=400, background_color="white", max_words=100
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{sentiment.capitalize()} Reviews")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_scattertext(self, output_html: str = "scattertext.html"):
        """Generate interactive scattertext visualization."""
        import scattertext as st

        corpus = self._build_corpus()

        html = st.produce_scattertext_explorer(
            corpus,
            category="positive",
            category_name="Positive",
            not_category_name="Negative",
            minimum_term_frequency=50,
            width_in_pixels=1000,
            transform=st.Scalers.log_scale_standardize,
        )

        Path(output_html).write_text(html)
        print(f"Saved to {output_html}")


# ==================== Part 3: PyTorch Dataset/DataLoader ====================

class IMDBDataset(Dataset):
    """
    PyTorch-compatible dataset for IMDB reviews.
    
    Works with pre-tokenized sequences from the _tokens column.
    Converts tokens to token IDs using a vocabulary mapping.
    For next token prediction: returns input sequence and target sequence (shifted by 1).
    """

    def __init__(
        self,
        tokens_list: list[list[str]],
        labels: list[int],
        vocab: dict[str, int],
        start_token: Optional[str] = None,
        end_token: Optional[str] = None,
    ):
        self.tokens_list = tokens_list
        self.labels = labels
        self.vocab = vocab
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = "<UNK>"
        self.unk_id = vocab.get(self.unk_token, 1)
        self.padding_idx = 0

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.

        Returns:
            dict with:
                - "token_ids": list[int] - token ID sequence (with optional start/end tokens)
                - "label": int - sentiment label (0=negative, 1=positive)
        """
        tokens = self.tokens_list[idx].copy()
        
        if self.start_token:
            tokens = [self.start_token] + tokens
        if self.end_token:
            tokens = tokens + [self.end_token]
        
        # Convert tokens to IDs
        token_ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        
        return {
            "token_ids": token_ids,
            "label": self.labels[idx],
        }


def collate_fn_for_lm(batch, padding_idx: int = 0, max_seq_length: Optional[int] = None):
    """
    Collate function for next token prediction (language modeling).
    Pads sequences and creates input/target pairs where target is shifted by 1.
    Memory-optimized version that truncates long sequences.
    
    Args:
        batch: List of dicts with "token_ids" and "label" keys
        padding_idx: Index to use for padding
        max_seq_length: Maximum sequence length (truncates longer sequences)
        
    Returns:
        dict with:
            - "input_ids": torch.Tensor (batch_size, max_seq_len)
            - "target_ids": torch.Tensor (batch_size, max_seq_len)
            - "label": torch.Tensor (batch_size,)
    """
    token_ids_list = [item["token_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Truncate sequences if max_seq_length is specified
    if max_seq_length is not None:
        token_ids_list = [seq[:max_seq_length] for seq in token_ids_list]
    
    # Find max length in batch (after truncation)
    max_len = max(len(seq) for seq in token_ids_list) if token_ids_list else 0
    
    if max_len == 0:
        # Empty batch
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "target_ids": torch.empty((0, 0), dtype=torch.long),
            "label": torch.empty((0,), dtype=torch.long),
        }
    
    batch_size = len(token_ids_list)
    
    # Pre-allocate tensors for efficiency
    input_ids = torch.full((batch_size, max_len), padding_idx, dtype=torch.long)
    target_ids = torch.full((batch_size, max_len), padding_idx, dtype=torch.long)
    
    # Fill tensors directly (more memory efficient than list appending)
    for i, token_ids in enumerate(token_ids_list):
        seq_len = len(token_ids)
        if seq_len > 1:
            # For next token prediction: input = [t0, t1, ..., tN-2], target = [t1, t2, ..., tN-1]
            # Copy sequences into pre-allocated tensors directly
            input_len = min(seq_len - 1, max_len)
            target_len = min(seq_len - 1, max_len)
            
            # Use direct assignment from list slices (more efficient)
            if input_len > 0:
                input_ids[i, :input_len] = torch.tensor(token_ids[:-1][:input_len], dtype=torch.long)
            if target_len > 0:
                target_ids[i, :target_len] = torch.tensor(token_ids[1:][:target_len], dtype=torch.long)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "label": labels_tensor,
    }


class IMDBDataModule:
    """
    Creates PyTorch Dataset and DataLoader instances for IMDB data.
    Supports vocabulary building and token-to-ID conversion for neural network training.
    
    Usage:
        preprocessor = TextPreprocessor("dataset/imdb-dataset.csv")
        df = preprocessor.load_data(remove_stopwords=True)
        train_df, val_df, test_df = preprocessor.get_splits()
        
        data_module = IMDBDataModule()
        data_module.build_vocab(train_df["_tokens"].tolist())
        train_dataset = data_module.get_torch_dataset(train_df, start_token="<s>", end_token="</s>")
        train_loader = data_module.get_torch_dataloader(train_df, batch_size=32, shuffle=True)
    """
    
    def __init__(self):
        self.vocab: Optional[dict[str, int]] = None
        self.padding_idx = 0

    def build_vocab(self, tokens_list: list[list[str]], min_freq: int = 1):
        """
        Build vocabulary from token sequences.
        
        Args:
            tokens_list: List of token sequences
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        from collections import Counter
        
        # Count token frequencies
        token_counter = Counter()
        for tokens in tokens_list:
            token_counter.update(tokens)
        
        # Build vocabulary: special tokens first, then frequent tokens
        self.vocab = {}
        # Add padding token
        self.vocab["<PAD>"] = 0
        
        # Add special tokens
        special_tokens = ["<UNK>", "<s>", "</s>"]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Add tokens that meet minimum frequency
        for token, freq in token_counter.items():
            if token not in self.vocab and freq >= min_freq:
                self.vocab[token] = len(self.vocab)
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def decode_sequence(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back to a string using the vocabulary.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed string
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        tokens = [id_to_token.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(tokens)
    

    def get_torch_dataset(
        self,
        df: pd.DataFrame,
        start_token: Optional[str] = None,
        end_token: Optional[str] = None,
    ) -> IMDBDataset:
        """
        Create a PyTorch Dataset from a DataFrame.

        Args:
            df: DataFrame (e.g., from TextPreprocessor.get_splits())
            start_token: optional token to prepend (e.g., '<s>')
            end_token: optional token to append (e.g., '</s>')

        Returns:
            IMDBDataset instance (compatible with torch.utils.data.Dataset)
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        return IMDBDataset(
            tokens_list=df["_tokens"].tolist(),
            labels=df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
            vocab=self.vocab,
            start_token=start_token,
            end_token=end_token,
        )

    def get_torch_dataloader(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        start_token: Optional[str] = None,
        end_token: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a DataFrame.

        Args:
            df: DataFrame (e.g., from TextPreprocessor.get_splits())
            batch_size: batch size for DataLoader (use smaller values for memory efficiency)
            shuffle: whether to shuffle the data
            start_token: optional token to prepend (e.g., '<s>')
            end_token: optional token to append (e.g., '</s>')
            max_seq_length: maximum sequence length (truncates longer sequences to save memory)
            num_workers: number of worker processes (0 = single process, safer for memory)

        Returns:
            PyTorch DataLoader instance with collate function for next token prediction
        """
        dataset = self.get_torch_dataset(df, start_token, end_token)
        collate_fn = functools.partial(
            collate_fn_for_lm, 
            padding_idx=self.padding_idx,
            max_seq_length=max_seq_length
        )
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False,  # Disable pin_memory to save RAM
        )


# ==================== Example Usage ====================

if __name__ == "__main__":
    import pprint

    # Part 1: Preprocessing
    preprocessor = TextPreprocessor("dataset/imdb-dataset.csv", sample_size=10000)
    df = preprocessor.load_data(remove_stopwords=True)

    # Part 2: Statistics
    stats_computer = IMDBDataStats(df)
    stats = stats_computer.get_stats()
    pprint.pprint(stats)

    # Part 3: PyTorch Dataset/DataLoader
    train_df, val_df, test_df = preprocessor.get_splits()

    # For neural network with PyTorch DataLoader
    data_module = IMDBDataModule()
    data_module.build_vocab(train_df["_tokens"].tolist())

    # Create dataset and dataloader with pre-tokenized sequences
    train_dataset = data_module.get_torch_dataset(train_df, start_token="<s>", end_token="</s>")
    train_loader = data_module.get_torch_dataloader(train_df, batch_size=32, shuffle=True)

    # Example: iterate through batches
    print(f"Dataset length: {len(train_dataset)}")
    print(f"First sample token_ids length: {len(train_dataset[0]['token_ids'])}")
    print(f"\nIterating through first batch:")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Target IDs shape: {batch['target_ids'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        break  # Just show first batch
