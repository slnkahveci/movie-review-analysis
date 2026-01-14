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

from src.data.stats import IMDBDataStats

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

    Note: For IMDBDataStats, use tokenizer_type="word" (default) to get the "_words" column.
    """

    def __init__(
        self,
        file_path: str,
        sample_size: Optional[int] = 50000,
        random_state: int = 42,
        tokenizer_type: Literal["word", "bpe", "wordpiece", "unigram"] = "word",
        tokenizer=None,
    ):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.random_state = random_state
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer

        self.data: Optional[pd.DataFrame] = None

        self.stop_words = set(stopwords.words("english"))
        if tokenizer_type == "word":
            self._tokenizer: Callable[[str], list[str]] = nltk.word_tokenize
        elif tokenizer_type == "bpe":
            self._tokenizer = RegexpTokenizer(r"\w+|[^\w\s]").tokenize
        elif tokenizer_type == "wordpiece":
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            ).tokenize
        elif tokenizer_type == "unigram":
            if tokenizer is None:
                raise ValueError(
                    "For 'unigram' tokenizer_type, a tokenizer instance must be provided."
                )
            self._tokenizer = tokenizer.tokenize

        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    def _preprocess_text(
        self,
        df: pd.DataFrame,
        remove_stopwords: bool = False,
        stemming: Literal["none", "stemming", "lemmatization"] = "none",
    ) -> pd.Series:
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
                lambda x: " ".join(
                    [word for word in x.split() if word not in self.stop_words]
                )
            )

        if self.tokenizer_type == "word":
            # Remove punctuation and numbers for word-based tokenization
            text_column = text_column.str.replace(r"[^\w\s]", "", regex=True)
        
        # stripping punctuation for other tokenizers might be harmful, so we skip it

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

    def load_data(
        self, remove_stopwords: bool = False, max_length: int = 512
    ) -> pd.DataFrame:
        # Load and preprocess the IMDB dataset.

        print(f"Loading data from {self.file_path}...")

        df = pd.read_csv(self.file_path)

        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=self.random_state)
            df = df.reset_index(drop=True)

        df["review"] = self._preprocess_text(df, remove_stopwords)

        # If tokenizer is provided and tokenizer_type is bpe, wordpiece, or unigram, encode texts
        if self.tokenizer and self.tokenizer_type in ["bpe", "wordpiece", "unigram"]:
            # Use truncation only, no padding (dynamic padding in DataLoader is much faster)
            encodings = self.tokenizer(
                df["review"].tolist(),
                truncation=True,
                padding=False,  # Dynamic padding per batch is much faster than padding all to max_length
                max_length=max_length,
            )
            df["_input_ids"] = encodings["input_ids"]
            df["_attention_mask"] = encodings["attention_mask"]
            df["_word_count"] = df["_input_ids"].apply(len)
        else:
            # Word-based tokenization saves to _words column
            df["_words"] = df["review"].apply(self._tokenize_words)
            df["_word_count"] = df["_words"].apply(len)

        # Length features
        df["_char_len"] = df["review"].str.len()

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

        print(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

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
        if "_words" in self.data.columns:
            save_df["_words_str"] = self.data["_words"].apply(" ".join)

        Dataset.from_pandas(save_df).save_to_disk(path)
        print(f"Saved to {path}")

    def load(self, path: str) -> pd.DataFrame:
        """Load processed data from disk."""
        from datasets import load_from_disk

        df = load_from_disk(path).to_pandas()

        # Reconstruct tokens if word-based
        if "_words_str" in df.columns:
            df["_words"] = df["_words_str"].str.split()
            df = df.drop(columns=["_words_str"])

        self.data = df

        print(f"Loaded {len(df)} reviews from {path}")
        return df


# ==================== Part 3: PyTorch Dataset/DataLoader ====================


class IMDBDataset(Dataset):
    """
    Simple PyTorch-compatible dataset for IMDB reviews.
    Stores raw tokens and labels; token-to-ID conversion happens in collate_fn.
    """

    def __init__(self, tokens_list: list[list[str]], labels: list[int]):
        self.tokens_list = tokens_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tokens": self.tokens_list[idx],
            "label": self.labels[idx],
        }


def collate_fn_for_lm(
    batch,
    vocab: dict[str, int],
    padding_idx: int = 0,
    max_seq_length: Optional[int] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    """
    Collate function that converts tokens to IDs and creates input/target pairs.
    """
    unk_id = vocab.get("<UNK>", 1)

    # Convert tokens to IDs with optional start/end tokens
    token_ids_list = []
    for item in batch:
        tokens = item["tokens"]
        if start_token:
            tokens = [start_token] + tokens
        if end_token:
            tokens = tokens + [end_token]
        token_ids = [vocab.get(t, unk_id) for t in tokens]
        token_ids_list.append(token_ids)

    labels = [item["label"] for item in batch]

    # Truncate BUT preserve end token
    if max_seq_length is not None:
        truncated = []
        for seq in token_ids_list:
            if len(seq) > max_seq_length:
                truncated.append(seq[: max_seq_length - 1] + [seq[-1]])
            else:
                truncated.append(seq)
        token_ids_list = truncated

    # Build input/target pairs
    inputs = []
    targets = []
    for seq in token_ids_list:
        if len(seq) > 1:
            inputs.append(seq[:-1])
            targets.append(seq[1:])
        else:
            inputs.append([padding_idx])
            targets.append([padding_idx])

    # Pad all sequences
    max_len = max(len(s) for s in inputs)
    input_ids = torch.full((len(inputs), max_len), padding_idx, dtype=torch.long)
    target_ids = torch.full((len(targets), max_len), padding_idx, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        target_ids[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "label": torch.tensor(labels, dtype=torch.long),
    }


class IMDBDataModule:
    """
    Creates PyTorch DataLoader instances for IMDB data.
    Handles vocabulary building and token-to-ID conversion.

    Usage:
        preprocessor = TextPreprocessor("dataset/imdb-dataset.csv")
        df = preprocessor.load_data(remove_stopwords=True)
        train_df, val_df, test_df = preprocessor.get_splits()

        data_module = IMDBDataModule()
        data_module.build_vocab(train_df["_words"].tolist())
        train_loader = data_module.get_dataloader(train_df, batch_size=32, shuffle=True)
    """

    def __init__(self):
        self.vocab: Optional[dict[str, int]] = None
        self.padding_idx = 0

    def build_vocab(self, tokens_list: list[list[str]], min_freq: int = 10):
        """
        Build vocabulary from token sequences.

        Args:
            tokens_list: List of token sequences
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        token_counter = Counter()
        for tokens in tokens_list:
            token_counter.update(tokens)

        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}

        for token, freq in token_counter.items():
            if token not in self.vocab and freq >= min_freq:
                self.vocab[token] = len(self.vocab)

        print(f"Built vocabulary with {len(self.vocab)} tokens")

    def decode_sequence(self, token_ids: list[int]) -> str:
        """Convert token IDs back to string."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        id_to_token = {idx: token for token, idx in self.vocab.items()}
        tokens = [id_to_token.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(tokens)

    def get_dataloader(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        start_token: Optional[str] = "<s>",
        end_token: Optional[str] = "</s>",
        max_seq_length: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a DataFrame.

        Args:
            df: DataFrame with "_words" column from TextPreprocessor
            batch_size: batch size
            shuffle: whether to shuffle
            start_token: token to prepend (default: "<s>")
            end_token: token to append (default: "</s>")
            max_seq_length: max sequence length (truncates longer sequences)
            num_workers: number of worker processes
            device: device string for pin_memory optimization

        Returns:
            PyTorch DataLoader
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        if "_words" not in df.columns:
            raise ValueError(
                "DataFrame must contain '_words' column. "
                "Use TextPreprocessor with tokenizer_type='word' (default)."
            )

        dataset = IMDBDataset(
            tokens_list=df["_words"].tolist(),
            labels=df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
        )

        collate_fn = functools.partial(
            collate_fn_for_lm,
            vocab=self.vocab,
            padding_idx=self.padding_idx,
            max_seq_length=max_seq_length,
            start_token=start_token,
            end_token=end_token,
        )

        pin_memory = device == "cuda" if device else False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# ==================== Example Usage ====================

if __name__ == "__main__":
    import pprint

    # Part 1: Preprocessing
    preprocessor = TextPreprocessor("dataset/imdb-dataset.csv", sample_size=10000)
    df = preprocessor.load_data(remove_stopwords=True)

    # Part 2: Statistics
    stats_computer = IMDBDataStats(df)
    stats = stats_computer.get_full_stats()
    pprint.pprint(stats)

    # Part 3: PyTorch DataLoader
    train_df, val_df, test_df = preprocessor.get_splits()

    data_module = IMDBDataModule()
    data_module.build_vocab(train_df["_words"].tolist())

    train_loader = data_module.get_dataloader(train_df, batch_size=32, shuffle=True)
    val_loader = data_module.get_dataloader(val_df, batch_size=32)

    # Example: iterate through batches
    print(f"\nDataLoader length: {len(train_loader)} batches")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Target IDs shape: {batch['target_ids'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        break
